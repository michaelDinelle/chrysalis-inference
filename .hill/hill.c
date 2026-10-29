#define _GNU_SOURCE
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <dirent.h>
#include <math.h>
#include <poll.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define ALPHA 0.05
#define BLOCKS 2
#define CALIBRATION_BLOCKS 6
#define HEADING "Rewards"

extern char **environ;

typedef struct {
    char *p;
    size_t n, cap;
} Buf;

static void grow(Buf *b, size_t add) {
    if (b->n + add + 1 > b->cap) {
        size_t c = b->cap ? b->cap : 256;
        while (c < b->n + add + 1) c *= 2;
        b->p = realloc(b->p, c);
        b->cap = c;
    }
}

static void bput(Buf *b, const char *s, size_t n) {
    grow(b, n);
    memcpy(b->p + b->n, s, n);
    b->n += n;
    b->p[b->n] = 0;
}

static void bputs(Buf *b, const char *s) { bput(b, s, strlen(s)); }

static void bputc(Buf *b, char c) { bput(b, &c, 1); }

static void vbprintf(Buf *b, const char *fmt, va_list ap) {
    va_list aq;
    va_copy(aq, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    grow(b, (size_t)n);
    vsnprintf(b->p + b->n, (size_t)n + 1, fmt, aq);
    va_end(aq);
    b->n += (size_t)n;
}

static void bprintf(Buf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vbprintf(b, fmt, ap);
    va_end(ap);
}

static char *bstr(Buf *b) {
    if (!b->p) grow(b, 0), b->p[0] = 0;
    return b->p;
}

static char *fmtstr(const char *fmt, ...) {
    Buf b = {0};
    va_list ap;
    va_start(ap, fmt);
    vbprintf(&b, fmt, ap);
    va_end(ap);
    return bstr(&b);
}

static jmp_buf stop_env;
static const char *stop_kind;
static char *stop_text;

static void stop(const char *kind, const char *fmt, ...) {
    Buf b = {0};
    va_list ap;
    va_start(ap, fmt);
    vbprintf(&b, fmt, ap);
    va_end(ap);
    stop_kind = kind;
    stop_text = bstr(&b);
    longjmp(stop_env, 1);
}

typedef enum { JNULL, JBOOL, JINT, JREAL, JSTR, JARR, JOBJ } JT;

typedef struct J {
    JT t;
    int b;
    long long i;
    double d;
    char *s;
    size_t len, cap;
    struct J **v;
    char **k;
} J;

static J *jnew(JT t) {
    J *j = calloc(1, sizeof *j);
    j->t = t;
    return j;
}

static J *jstr(const char *s) {
    J *j = jnew(JSTR);
    j->s = strdup(s);
    return j;
}

static J *jreal(double d) {
    J *j = jnew(JREAL);
    j->d = d;
    return j;
}

static J *jget(const J *o, const char *key) {
    if (!o || o->t != JOBJ) return NULL;
    for (size_t i = 0; i < o->len; i++)
        if (!strcmp(o->k[i], key)) return o->v[i];
    return NULL;
}

static void jset(J *o, const char *key, J *val) {
    for (size_t i = 0; o->t == JOBJ && i < o->len; i++)
        if (!strcmp(o->k[i], key)) {
            o->v[i] = val;
            return;
        }
    if (o->len == o->cap) {
        o->cap = o->cap ? 2 * o->cap : 8;
        o->v = realloc(o->v, o->cap * sizeof *o->v);
        if (o->t == JOBJ) o->k = realloc(o->k, o->cap * sizeof *o->k);
    }
    if (o->t == JOBJ) o->k[o->len] = strdup(key);
    o->v[o->len++] = val;
}

static void jpush(J *a, J *val) { jset(a, NULL, val); }

static const char *jtext(const J *j) { return j && j->t == JSTR ? j->s : NULL; }

static const char *jtext_or(const J *o, const char *key, const char *fallback) {
    const char *s = jtext(jget(o, key));
    return s ? s : fallback;
}

static int jnum(const J *j) { return j && (j->t == JINT || j->t == JREAL); }

static double jd(const J *j) { return j->t == JINT ? (double)j->i : j->d; }

static int jtruthy(const J *j) {
    if (!j) return 0;
    switch (j->t) {
    case JNULL: return 0;
    case JBOOL: return j->b;
    case JINT: return j->i != 0;
    case JREAL: return j->d != 0;
    case JSTR: return j->s[0] != 0;
    default: return j->len != 0;
    }
}

static long long jint_or(const J *o, const char *key, long long fallback) {
    J *v = jget(o, key);
    if (!v || v->t == JNULL || v->t == JBOOL) return fallback;
    return v->t == JINT ? v->i : (long long)v->d;
}

typedef struct {
    const char *p, *end;
    int bad;
} Parser;

static void skipws(Parser *p) {
    while (p->p < p->end && (*p->p == ' ' || *p->p == '\t' || *p->p == '\n' || *p->p == '\r')) p->p++;
}

static void utf8(Buf *b, unsigned c) {
    if (c < 0x80) bputc(b, (char)c);
    else if (c < 0x800) bputc(b, (char)(0xc0 | c >> 6)), bputc(b, (char)(0x80 | (c & 0x3f)));
    else if (c < 0x10000) bputc(b, (char)(0xe0 | c >> 12)), bputc(b, (char)(0x80 | (c >> 6 & 0x3f))), bputc(b, (char)(0x80 | (c & 0x3f)));
    else bputc(b, (char)(0xf0 | c >> 18)), bputc(b, (char)(0x80 | (c >> 12 & 0x3f))), bputc(b, (char)(0x80 | (c >> 6 & 0x3f))), bputc(b, (char)(0x80 | (c & 0x3f)));
}

static int hex4(Parser *p, unsigned *out) {
    if (p->end - p->p < 4) return 0;
    unsigned v = 0;
    for (int i = 0; i < 4; i++) {
        char c = p->p[i];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= (unsigned)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (unsigned)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (unsigned)(c - 'A' + 10);
        else return 0;
    }
    p->p += 4;
    *out = v;
    return 1;
}

static char *parse_string(Parser *p) {
    Buf b = {0};
    p->p++;
    while (p->p < p->end && *p->p != '"') {
        unsigned char c = (unsigned char)*p->p;
        if (c < 0x20) {
            p->bad = 1;
            return bstr(&b);
        }
        if (c != '\\') {
            bputc(&b, (char)c);
            p->p++;
            continue;
        }
        p->p++;
        if (p->p >= p->end) break;
        char e = *p->p++;
        switch (e) {
        case '"': bputc(&b, '"'); break;
        case '\\': bputc(&b, '\\'); break;
        case '/': bputc(&b, '/'); break;
        case 'b': bputc(&b, '\b'); break;
        case 'f': bputc(&b, '\f'); break;
        case 'n': bputc(&b, '\n'); break;
        case 'r': bputc(&b, '\r'); break;
        case 't': bputc(&b, '\t'); break;
        case 'u': {
            unsigned u, lo;
            if (!hex4(p, &u)) {
                p->bad = 1;
                return bstr(&b);
            }
            if (u >= 0xd800 && u < 0xdc00 && p->end - p->p >= 6 && p->p[0] == '\\' && p->p[1] == 'u') {
                Parser q = *p;
                q.p += 2;
                if (hex4(&q, &lo) && lo >= 0xdc00 && lo < 0xe000) {
                    *p = q;
                    u = 0x10000 + ((u - 0xd800) << 10) + (lo - 0xdc00);
                }
            }
            utf8(&b, u);
            break;
        }
        default: p->bad = 1; return bstr(&b);
        }
    }
    if (p->p >= p->end) p->bad = 1;
    else p->p++;
    return bstr(&b);
}

static J *parse_value(Parser *p);

static J *parse_number(Parser *p) {
    const char *s = p->p;
    int real = 0;
    if (p->p < p->end && *p->p == '-') p->p++;
    if (p->end - p->p >= 8 && !strncmp(p->p, "Infinity", 8)) {
        p->p += 8;
        return jreal(*s == '-' ? -INFINITY : INFINITY);
    }
    while (p->p < p->end && isdigit((unsigned char)*p->p)) p->p++;
    if (p->p < p->end && *p->p == '.') {
        real = 1;
        p->p++;
        while (p->p < p->end && isdigit((unsigned char)*p->p)) p->p++;
    }
    if (p->p < p->end && (*p->p == 'e' || *p->p == 'E')) {
        real = 1;
        p->p++;
        if (p->p < p->end && (*p->p == '+' || *p->p == '-')) p->p++;
        while (p->p < p->end && isdigit((unsigned char)*p->p)) p->p++;
    }
    char *tok = strndup(s, (size_t)(p->p - s));
    if (p->p == s || (tok[0] == '-' && !tok[1])) p->bad = 1;
    if (!real) {
        errno = 0;
        long long v = strtoll(tok, NULL, 10);
        if (!errno) {
            J *j = jnew(JINT);
            j->i = v;
            return j;
        }
    }
    return jreal(strtod(tok, NULL));
}

static J *parse_value(Parser *p) {
    skipws(p);
    if (p->p >= p->end) {
        p->bad = 1;
        return jnew(JNULL);
    }
    char c = *p->p;
    if (c == '{') {
        J *o = jnew(JOBJ);
        p->p++;
        skipws(p);
        if (p->p < p->end && *p->p == '}') {
            p->p++;
            return o;
        }
        while (!p->bad) {
            skipws(p);
            if (p->p >= p->end || *p->p != '"') {
                p->bad = 1;
                break;
            }
            char *key = parse_string(p);
            skipws(p);
            if (p->p >= p->end || *p->p != ':') {
                p->bad = 1;
                break;
            }
            p->p++;
            jset(o, key, parse_value(p));
            skipws(p);
            if (p->p < p->end && *p->p == ',') {
                p->p++;
                continue;
            }
            if (p->p < p->end && *p->p == '}') {
                p->p++;
                break;
            }
            p->bad = 1;
        }
        return o;
    }
    if (c == '[') {
        J *a = jnew(JARR);
        p->p++;
        skipws(p);
        if (p->p < p->end && *p->p == ']') {
            p->p++;
            return a;
        }
        while (!p->bad) {
            jpush(a, parse_value(p));
            skipws(p);
            if (p->p < p->end && *p->p == ',') {
                p->p++;
                continue;
            }
            if (p->p < p->end && *p->p == ']') {
                p->p++;
                break;
            }
            p->bad = 1;
        }
        return a;
    }
    if (c == '"') {
        J *j = jnew(JSTR);
        j->s = parse_string(p);
        return j;
    }
    if (p->end - p->p >= 4 && !strncmp(p->p, "true", 4)) {
        p->p += 4;
        J *j = jnew(JBOOL);
        j->b = 1;
        return j;
    }
    if (p->end - p->p >= 5 && !strncmp(p->p, "false", 5)) {
        p->p += 5;
        return jnew(JBOOL);
    }
    if (p->end - p->p >= 4 && !strncmp(p->p, "null", 4)) {
        p->p += 4;
        return jnew(JNULL);
    }
    if (p->end - p->p >= 3 && !strncmp(p->p, "NaN", 3)) {
        p->p += 3;
        return jreal(NAN);
    }
    if (c == '-' || isdigit((unsigned char)c) || c == 'I') return parse_number(p);
    p->bad = 1;
    return jnew(JNULL);
}

static J *parse(const char *s, size_t n) {
    Parser p = {s, s + n, 0};
    J *j = parse_value(&p);
    skipws(&p);
    if (p.bad || p.p != p.end) return NULL;
    return j;
}

static void pyfloat(Buf *b, double d) {
    if (isnan(d)) {
        bputs(b, "NaN");
        return;
    }
    if (isinf(d)) {
        bputs(b, d > 0 ? "Infinity" : "-Infinity");
        return;
    }
    if (d == 0) {
        bputs(b, signbit(d) ? "-0.0" : "0.0");
        return;
    }
    char tmp[64];
    for (int prec = 1; prec <= 17; prec++) {
        snprintf(tmp, sizeof tmp, "%.*e", prec - 1, d);
        if (strtod(tmp, NULL) == d) break;
    }
    char digits[32];
    int nd = 0, neg = tmp[0] == '-';
    const char *q = tmp + neg;
    for (; *q && *q != 'e'; q++)
        if (isdigit((unsigned char)*q)) digits[nd++] = *q;
    int decpt = atoi(q + 1) + 1;
    while (nd > 1 && digits[nd - 1] == '0') nd--;
    if (neg) bputc(b, '-');
    if (decpt <= -4 || decpt > 16) {
        bputc(b, digits[0]);
        if (nd > 1) bputc(b, '.'), bput(b, digits + 1, (size_t)nd - 1);
        int e = decpt - 1;
        bprintf(b, "e%c%02d", e < 0 ? '-' : '+', abs(e));
    } else if (decpt <= 0) {
        bputs(b, "0.");
        for (int i = 0; i < -decpt; i++) bputc(b, '0');
        bput(b, digits, (size_t)nd);
    } else if (decpt >= nd) {
        bput(b, digits, (size_t)nd);
        for (int i = nd; i < decpt; i++) bputc(b, '0');
        bputs(b, ".0");
    } else {
        bput(b, digits, (size_t)decpt);
        bputc(b, '.');
        bput(b, digits + decpt, (size_t)(nd - decpt));
    }
}

static void dump_str(Buf *b, const char *s) {
    bputc(b, '"');
    const unsigned char *u = (const unsigned char *)s;
    while (*u) {
        unsigned c = *u;
        if (c < 0x80) {
            switch (c) {
            case '"': bputs(b, "\\\""); break;
            case '\\': bputs(b, "\\\\"); break;
            case '\n': bputs(b, "\\n"); break;
            case '\r': bputs(b, "\\r"); break;
            case '\t': bputs(b, "\\t"); break;
            case '\b': bputs(b, "\\b"); break;
            case '\f': bputs(b, "\\f"); break;
            default:
                if (c < 0x20) bprintf(b, "\\u%04x", c);
                else bputc(b, (char)c);
            }
            u++;
            continue;
        }
        unsigned cp;
        int extra;
        if ((c & 0xe0) == 0xc0) cp = c & 0x1f, extra = 1;
        else if ((c & 0xf0) == 0xe0) cp = c & 0x0f, extra = 2;
        else cp = c & 0x07, extra = 3;
        u++;
        for (int i = 0; i < extra && (*u & 0xc0) == 0x80; i++) cp = cp << 6 | (*u++ & 0x3f);
        if (cp >= 0x10000) {
            cp -= 0x10000;
            bprintf(b, "\\u%04x\\u%04x", 0xd800 + (cp >> 10), 0xdc00 + (cp & 0x3ff));
        } else
            bprintf(b, "\\u%04x", cp);
    }
    bputc(b, '"');
}

static const char **sort_keys_k;

static int key_cmp(const void *a, const void *b) {
    return strcmp(sort_keys_k[*(const size_t *)a], sort_keys_k[*(const size_t *)b]);
}

static void dump(Buf *b, const J *j) {
    switch (j->t) {
    case JNULL: bputs(b, "null"); break;
    case JBOOL: bputs(b, j->b ? "true" : "false"); break;
    case JINT: bprintf(b, "%lld", j->i); break;
    case JREAL: pyfloat(b, j->d); break;
    case JSTR: dump_str(b, j->s); break;
    case JARR:
        bputc(b, '[');
        for (size_t i = 0; i < j->len; i++) {
            if (i) bputs(b, ", ");
            dump(b, j->v[i]);
        }
        bputc(b, ']');
        break;
    case JOBJ: {
        size_t *idx = malloc((j->len + 1) * sizeof *idx);
        for (size_t i = 0; i < j->len; i++) idx[i] = i;
        sort_keys_k = (const char **)j->k;
        qsort(idx, j->len, sizeof *idx, key_cmp);
        bputc(b, '{');
        for (size_t i = 0; i < j->len; i++) {
            if (i) bputs(b, ", ");
            dump_str(b, j->k[idx[i]]);
            bputs(b, ": ");
            dump(b, j->v[idx[i]]);
        }
        bputc(b, '}');
        free(idx);
        break;
    }
    }
}

static const uint32_t SHA_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be,
    0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa,
    0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85,
    0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
    0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

#define ROR(x, n) ((x) >> (n) | (x) << (32 - (n)))

static void sha_block(uint32_t h[8], const uint8_t *p) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 | (uint32_t)p[4 * i + 2] << 8 | p[4 * i + 3];
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = ROR(w[i - 15], 7) ^ ROR(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = ROR(w[i - 2], 17) ^ ROR(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], k = h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = k + (ROR(e, 6) ^ ROR(e, 11) ^ ROR(e, 25)) + ((e & f) ^ (~e & g)) + SHA_K[i] + w[i];
        uint32_t t2 = (ROR(a, 2) ^ ROR(a, 13) ^ ROR(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        k = g, g = f, f = e, e = d + t1, d = c, c = b, b = a, a = t1 + t2;
    }
    h[0] += a, h[1] += b, h[2] += c, h[3] += d, h[4] += e, h[5] += f, h[6] += g, h[7] += k;
}

static char *sha16(const char *s, size_t n) {
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    size_t i = 0;
    for (; i + 64 <= n; i += 64) sha_block(h, (const uint8_t *)s + i);
    uint8_t tail[128] = {0};
    size_t rest = n - i;
    memcpy(tail, s + i, rest);
    tail[rest] = 0x80;
    size_t total = rest + 1 + 8 <= 64 ? 64 : 128;
    uint64_t bits = (uint64_t)n * 8;
    for (int k = 0; k < 8; k++) tail[total - 1 - k] = (uint8_t)(bits >> (8 * k));
    sha_block(h, tail);
    if (total == 128) sha_block(h, tail + 64);
    char *out = malloc(17);
    for (int k = 0; k < 8; k++) snprintf(out + 2 * k, 3, "%02x", (h[k / 4] >> (24 - 8 * (k % 4))) & 0xff);
    out[16] = 0;
    return out;
}

static char *digest_json(const J *j) {
    Buf b = {0};
    dump(&b, j);
    return sha16(b.p, b.n);
}

typedef struct {
    int status;
    Buf out, err;
} Proc;

static Proc spawn(char *const argv[], const char *cwd, const char *in, size_t inlen, int capture_out, int capture_err, int out_fd,
                  char *const *env) {
    Proc r = {0};
    int pin[2] = {-1, -1}, pout[2] = {-1, -1}, perr[2] = {-1, -1};
    if (in && pipe2(pin, O_CLOEXEC)) stop("ERROR", "pipe: %s", strerror(errno));
    if (capture_out && pipe2(pout, O_CLOEXEC)) stop("ERROR", "pipe: %s", strerror(errno));
    if (capture_err && pipe2(perr, O_CLOEXEC)) stop("ERROR", "pipe: %s", strerror(errno));
    fflush(stdout);
    fflush(stderr);
    pid_t pid = fork();
    if (pid < 0) stop("ERROR", "fork: %s", strerror(errno));
    if (!pid) {
        if (cwd && chdir(cwd)) _exit(127);
        if (in) dup2(pin[0], 0);
        else {
            int nul = open("/dev/null", O_RDONLY);
            dup2(nul, 0);
        }
        if (capture_out) dup2(pout[1], 1);
        else if (out_fd >= 0) dup2(out_fd, 1);
        if (capture_err) dup2(perr[1], 2);
        else if (out_fd >= 0 && !capture_out) dup2(out_fd, 2);
        if (env) environ = (char **)env;
        execvp(argv[0], argv);
        _exit(127);
    }
    if (in) close(pin[0]);
    if (capture_out) close(pout[1]);
    if (capture_err) close(perr[1]);
    size_t sent = 0;
    if (in && !inlen) close(pin[1]), in = NULL;
    if (in) fcntl(pin[1], F_SETFL, O_NONBLOCK);
    int open_out = capture_out, open_err = capture_err;
    char chunk[65536];
    while (in || open_out || open_err) {
        struct pollfd fds[3];
        int nf = 0, iw = -1, io = -1, ie = -1;
        if (in) iw = nf, fds[nf++] = (struct pollfd){pin[1], POLLOUT, 0};
        if (open_out) io = nf, fds[nf++] = (struct pollfd){pout[0], POLLIN, 0};
        if (open_err) ie = nf, fds[nf++] = (struct pollfd){perr[0], POLLIN, 0};
        if (poll(fds, (nfds_t)nf, -1) < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (iw >= 0 && fds[iw].revents) {
            ssize_t w = write(pin[1], in + sent, inlen - sent);
            if (w > 0) sent += (size_t)w;
            if (w < 0 && errno != EAGAIN) sent = inlen;
            if (sent >= inlen) close(pin[1]), in = NULL;
        }
        if (io >= 0 && fds[io].revents) {
            ssize_t got = read(pout[0], chunk, sizeof chunk);
            if (got > 0) bput(&r.out, chunk, (size_t)got);
            else close(pout[0]), open_out = 0;
        }
        if (ie >= 0 && fds[ie].revents) {
            ssize_t got = read(perr[0], chunk, sizeof chunk);
            if (got > 0) bput(&r.err, chunk, (size_t)got);
            else close(perr[0]), open_err = 0;
        }
    }
    int st;
    while (waitpid(pid, &st, 0) < 0 && errno == EINTR) {
    }
    r.status = WIFEXITED(st) ? WEXITSTATUS(st) : -WTERMSIG(st);
    bstr(&r.out);
    bstr(&r.err);
    return r;
}

static char *strip(char *s) {
    while (*s && isspace((unsigned char)*s)) s++;
    size_t n = strlen(s);
    while (n && isspace((unsigned char)s[n - 1])) s[--n] = 0;
    return s;
}

static char *root_dir;

static Buf git_buf(const char *in, size_t inlen, const char **args) {
    char *argv[64];
    int n = 0;
    argv[n++] = "git";
    for (; args[n - 1]; n++) argv[n] = (char *)args[n - 1];
    argv[n] = NULL;
    Proc r = spawn(argv, root_dir, in, inlen, 1, 1, -1, NULL);
    if (r.status) {
        Buf j = {0};
        for (int i = 0; argv[i]; i++) {
            if (i) bputc(&j, ' ');
            bputs(&j, argv[i]);
        }
        stop("ERROR", "%s: %s", bstr(&j), strip(r.err.p));
    }
    return r.out;
}

static char *git_args(const char *in, size_t inlen, const char **args) { return git_buf(in, inlen, args).p; }

#define GIT(...) git_args(NULL, 0, (const char *[]){__VA_ARGS__, NULL})

typedef struct {
    char **v;
    size_t n, cap;
} Strs;

static void spush(Strs *s, char *x) {
    if (s->n == s->cap) s->cap = s->cap ? 2 * s->cap : 16, s->v = realloc(s->v, s->cap * sizeof *s->v);
    s->v[s->n++] = x;
}

static Strs zsplit(char *s, size_t n) {
    Strs out = {0};
    size_t i = 0;
    while (i < n) {
        char *e = memchr(s + i, 0, n - i);
        size_t len = e ? (size_t)(e - (s + i)) : n - i;
        if (len) spush(&out, s + i);
        i += len + 1;
    }
    return out;
}

typedef struct {
    char *path, *sha;
} Pair;

typedef struct {
    Pair *v;
    size_t n, cap;
} Pairs;

static void ppush(Pairs *p, char *path, char *sha) {
    if (p->n == p->cap) p->cap = p->cap ? 2 * p->cap : 64, p->v = realloc(p->v, p->cap * sizeof *p->v);
    p->v[p->n++] = (Pair){path, sha};
}

static int pair_cmp(const void *a, const void *b) {
    const Pair *x = a, *y = b;
    int c = strcmp(x->path, y->path);
    return c ? c : strcmp(x->sha, y->sha);
}

static int str_cmp(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }

static int is_file(const char *path) {
    struct stat st;
    char *full = fmtstr("%s/%s", root_dir, path);
    int ok = !stat(full, &st) && S_ISREG(st.st_mode);
    free(full);
    return ok;
}

static int plain(const char *spec) {
    return !strpbrk(spec, "*?[\\") && spec[0] != ':';
}

static int whole(const char *spec) { return !spec[0] || !strcmp(spec, ".") || !strcmp(spec, "./"); }

static int under1(const char *path, const char *spec) {
    if (whole(spec)) return 1;
    size_t n = strlen(spec);
    while (n && spec[n - 1] == '/') n--;
    return !strncmp(path, spec, n) && (path[n] == 0 || path[n] == '/');
}

static int under(const char *path, Strs *specs) {
    for (size_t i = 0; i < specs->n; i++)
        if (under1(path, specs->v[i])) return 1;
    return 0;
}

static int words(const char *s, char out[][64], int want) {
    int n = 0;
    while (n < want) {
        while (*s == ' ') s++;
        if (!*s || *s == '\t') break;
        size_t len = strcspn(s, " \t");
        size_t keep = len < 63 ? len : 63;
        memcpy(out[n], s, keep);
        out[n++][keep] = 0;
        s += len;
    }
    return n;
}

static void hash_into(Pairs *out, Strs *paths) {
    if (!paths->n) return;
    Buf in = {0};
    for (size_t i = 0; i < paths->n; i++) bputs(&in, paths->v[i]), bputc(&in, '\n');
    char *shas = git_args(in.p, in.n, (const char *[]){"hash-object", "--stdin-paths", NULL});
    char *save = NULL, *tok = strtok_r(shas, " \t\n", &save);
    for (size_t i = 0; i < paths->n && tok; i++, tok = strtok_r(NULL, " \t\n", &save)) ppush(out, paths->v[i], strdup(tok));
}

static void add_paths(const char **args, int *n, Strs *paths) {
    for (size_t i = 0; i < paths->n; i++) args[(*n)++] = paths->v[i];
    args[*n] = NULL;
}

static Pairs listing(const char *tree, Strs *paths) {
    Pairs out = {0};
    if (!paths->n) return out;
    const char *args[80];
    int n = 0;
    if (!strcmp(tree, "worktree")) {
        args[n++] = "ls-files", args[n++] = "-z", args[n++] = "-s", args[n++] = "-v", args[n++] = "--";
        add_paths(args, &n, paths);
        Buf idx = git_buf(NULL, 0, args);
        n = 0;
        args[n++] = "diff-files", args[n++] = "-z", args[n++] = "--name-only", args[n++] = "--";
        add_paths(args, &n, paths);
        Buf dirty = git_buf(NULL, 0, args);
        n = 0;
        args[n++] = "ls-files", args[n++] = "-z", args[n++] = "-o", args[n++] = "--exclude-standard", args[n++] = "--";
        add_paths(args, &n, paths);
        Buf untracked = git_buf(NULL, 0, args);
        Strs changed = zsplit(dirty.p ? dirty.p : "", dirty.n);
        qsort(changed.v, changed.n, sizeof *changed.v, str_cmp);
        Strs hash = {0};
        Strs entries = zsplit(idx.p ? idx.p : "", idx.n);
        char *last = NULL;
        for (size_t i = 0; i < entries.n; i++) {
            char *e = entries.v[i];
            char *tab = strchr(e, '\t');
            if (!tab) continue;
            char *path = tab + 1;
            char tag = e[0];
            char w[2][64] = {{0}};
            words(e + 2, w, 2);
            const char *mode = w[0], *sha = w[1];
            if (last && !strcmp(last, path)) continue;
            last = path;
            if (!strcmp(mode, "160000")) continue;
            int dirty_path = bsearch(&path, changed.v, changed.n, sizeof *changed.v, str_cmp) != NULL;
            int needs = dirty_path || !strcmp(mode, "120000") || islower((unsigned char)tag) || tag == 'S';
            if (needs) {
                if (is_file(path)) spush(&hash, path);
            } else
                ppush(&out, path, strdup(sha));
        }
        Strs fresh = zsplit(untracked.p ? untracked.p : "", untracked.n);
        for (size_t i = 0; i < fresh.n; i++)
            if (is_file(fresh.v[i])) spush(&hash, fresh.v[i]);
        hash_into(&out, &hash);
    } else if (!strcmp(tree, "index")) {
        args[n++] = "ls-files", args[n++] = "-z", args[n++] = "-s", args[n++] = "--";
        add_paths(args, &n, paths);
        Buf b = git_buf(NULL, 0, args);
        Strs entries = zsplit(b.p ? b.p : "", b.n);
        for (size_t i = 0; i < entries.n; i++) {
            char *tab = strchr(entries.v[i], '\t');
            char w[2][64];
            if (tab && words(entries.v[i], w, 2) == 2) ppush(&out, tab + 1, strdup(w[1]));
        }
    } else {
        args[n++] = "ls-tree", args[n++] = "-z", args[n++] = "-r", args[n++] = tree, args[n++] = "--";
        add_paths(args, &n, paths);
        Buf b = git_buf(NULL, 0, args);
        Strs entries = zsplit(b.p ? b.p : "", b.n);
        for (size_t i = 0; i < entries.n; i++) {
            char *tab = strchr(entries.v[i], '\t');
            char w[3][64];
            if (tab && words(entries.v[i], w, 3) == 3) ppush(&out, tab + 1, strdup(w[2]));
        }
    }
    qsort(out.v, out.n, sizeof *out.v, pair_cmp);
    return out;
}

static char *pairs_digest(Pairs *p) {
    Buf b = {0};
    bputc(&b, '[');
    for (size_t i = 0; i < p->n; i++) {
        if (i) bputs(&b, ", ");
        bputc(&b, '[');
        dump_str(&b, p->v[i].path);
        bputs(&b, ", ");
        dump_str(&b, p->v[i].sha);
        bputc(&b, ']');
    }
    bputc(&b, ']');
    return sha16(b.p, b.n);
}

typedef struct {
    J *spec;
    Strs measured, harness;
} Spec;

static Spec load_spec(const char *text, size_t n) {
    Spec s = {0};
    s.spec = parse(text, n);
    if (!s.spec || s.spec->t != JOBJ) stop("ERROR", "hill.json is not valid JSON");
    J *rewards = jget(s.spec, "rewards"), *instruments = jget(s.spec, "instruments");
    if (!rewards || rewards->t != JOBJ || !instruments || instruments->t != JOBJ) stop("ERROR", "hill.json needs rewards and instruments");
    for (size_t i = 0; i < rewards->len; i++) {
        J *r = rewards->v[i];
        const char *better = jtext(jget(r, "better"));
        if (!better || (strcmp(better, "higher") && strcmp(better, "lower")))
            stop("ERROR", "reward %s: better must be higher or lower", rewards->k[i]);
        const char *inst = jtext(jget(r, "instrument"));
        if (!inst || !jget(instruments, inst)) {
            Buf b = {0};
            J *named = jget(r, "instrument");
            if (!named) bputs(&b, "None");
            else if (named->t == JSTR) bputs(&b, named->s);
            else dump(&b, named);
            stop("ERROR", "reward %s: unknown instrument %s", rewards->k[i], bstr(&b));
        }
    }
    J *m = jget(s.spec, "measured");
    for (size_t i = 0; m && m->t == JARR && i < m->len; i++) spush(&s.measured, m->v[i]->s);
    Strs h = {0};
    J *hp = jget(s.spec, "harness");
    for (size_t i = 0; hp && hp->t == JARR && i < hp->len; i++) spush(&h, hp->v[i]->s);
    spush(&h, "hill.json");
    spush(&h, ".hill/hill.c");
    spush(&h, ".hill/hill");
    for (size_t i = 0; i < h.n; i++)
        for (size_t k = i + 1; k < h.n; k++)
            if (strcmp(h.v[k], h.v[i]) < 0) {
                char *t = h.v[i];
                h.v[i] = h.v[k], h.v[k] = t;
            }
    for (size_t i = 0; i < h.n; i++)
        if (!i || strcmp(h.v[i], h.v[i - 1])) spush(&s.harness, h.v[i]);
    return s;
}

static char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    Buf b = {0};
    char chunk[65536];
    size_t got;
    while ((got = fread(chunk, 1, sizeof chunk, f))) bput(&b, chunk, got);
    fclose(f);
    if (len) *len = b.n;
    return bstr(&b);
}

static Spec load_spec_file(void) {
    size_t n;
    char *path = fmtstr("%s/hill.json", root_dir);
    char *text = read_file(path, &n);
    if (!text) stop("ERROR", "no hill.json at the repository root");
    return load_spec(text, n);
}

typedef struct {
    char *product, *harness;
} Digests;

static int all_plain(Spec *s) {
    int fast = 1;
    for (size_t i = 0; i < s->measured.n; i++) fast &= plain(s->measured.v[i]);
    for (size_t i = 0; i < s->harness.n; i++) fast &= plain(s->harness.v[i]);
    return fast;
}

static Digests split_digests(Spec *s, Pairs *everything) {
    Pairs product = {0}, harness = {0};
    for (size_t i = 0; i < everything->n; i++) {
        char *p = everything->v[i].path;
        if (under(p, &s->harness)) ppush(&harness, p, everything->v[i].sha);
        if (under(p, &s->measured) && !under(p, &s->harness)) ppush(&product, p, everything->v[i].sha);
    }
    return (Digests){pairs_digest(&product), pairs_digest(&harness)};
}

static Digests digests(Spec *s, const char *tree) {
    Digests d;
    Pairs product = {0}, harness = {0};
    if (all_plain(s)) {
        Strs all = {0};
        for (size_t i = 0; i < s->measured.n; i++) spush(&all, s->measured.v[i]);
        for (size_t i = 0; i < s->harness.n; i++) spush(&all, s->harness.v[i]);
        Pairs everything = listing(tree, &all);
        return split_digests(s, &everything);
    } else {
        Pairs m = listing(tree, &s->measured);
        for (size_t i = 0; i < m.n; i++)
            if (!under(m.v[i].path, &s->harness)) ppush(&product, m.v[i].path, m.v[i].sha);
        harness = listing(tree, &s->harness);
    }
    d.product = pairs_digest(&product);
    d.harness = pairs_digest(&harness);
    return d;
}

static Pairs files_of(Spec *s) {
    Strs all = {0};
    for (size_t i = 0; i < s->measured.n; i++) spush(&all, s->measured.v[i]);
    for (size_t i = 0; i < s->harness.n; i++) spush(&all, s->harness.v[i]);
    return listing("worktree", &all);
}

typedef struct {
    char *ok[8];
    size_t len[8];
} Blobs;

static Blobs cat_files(const char **names, int n) {
    Blobs out = {0};
    Buf in = {0};
    for (int i = 0; i < n; i++) bputs(&in, names[i]), bputc(&in, '\n');
    char *argv[] = {"git", "cat-file", "--batch", NULL};
    Proc r = spawn(argv, root_dir, in.p, in.n, 1, 1, -1, NULL);
    if (r.status) stop("ERROR", "git cat-file --batch: %s", strip(r.err.p));
    char *p = r.out.p, *end = r.out.p + r.out.n;
    for (int i = 0; i < n && p < end; i++) {
        char *nl = memchr(p, '\n', (size_t)(end - p));
        if (!nl) break;
        *nl = 0;
        char *header = p;
        p = nl + 1;
        char *sp = strrchr(header, ' ');
        if (strstr(header, " missing") && (!sp || !strcmp(sp, " missing"))) continue;
        if (!sp) continue;
        size_t size = strtoull(sp + 1, NULL, 10);
        if ((size_t)(end - p) < size) break;
        char *type = strchr(header, ' ');
        if (type && !strncmp(type + 1, "commit", 6)) {
            out.ok[i] = strndup(header, (size_t)(type - header));
            out.len[i] = strlen(out.ok[i]);
        } else {
            out.ok[i] = strndup(p, size);
            out.len[i] = size;
        }
        p += size + 1;
    }
    return out;
}

static char *head_commit(void) {
    const char *names[] = {"HEAD"};
    return cat_files(names, 1).ok[0];
}

typedef struct {
    J **v;
    size_t n;
} Ledger;

static Ledger read_ledger_text(const char *text, size_t n) {
    Ledger l = {0};
    size_t cap = 0;
    const char *p = text, *end = text + n;
    while (p < end) {
        const char *nl = memchr(p, '\n', (size_t)(end - p));
        const char *e = nl ? nl : end;
        const char *q = p;
        while (q < e && isspace((unsigned char)*q)) q++;
        if (q < e) {
            J *j = parse(p, (size_t)(e - p));
            if (!j || j->t != JOBJ) stop("ERROR", ".hill/ledger.jsonl has a line that is not valid JSON");
            if (l.n == cap) cap = cap ? 2 * cap : 64, l.v = realloc(l.v, cap * sizeof *l.v);
            l.v[l.n++] = j;
        }
        p = e + 1;
    }
    return l;
}

static Ledger read_ledger(void) {
    size_t n = 0;
    char *path = fmtstr("%s/.hill/ledger.jsonl", root_dir);
    char *text = read_file(path, &n);
    return read_ledger_text(text ? text : "", text ? n : 0);
}

static int is(const J *e, const char *key, const char *value) {
    const char *s = jtext(jget(e, key));
    return s && !strcmp(s, value);
}

static void default_file(char *path, const char *text) {
    FILE *f = access(path, F_OK) ? fopen(path, "w") : NULL;
    if (f) fputs(text, f), fclose(f);
}

static char *hill_dir(void) {
    char *d = fmtstr("%s/.hill", root_dir);
    mkdir(d, 0777);
    default_file(fmtstr("%s/.gitignore", d), "trees/\nlogs/\nlock\nbin.path\n");
    default_file(fmtstr("%s/.gitattributes", d), "ledger.jsonl merge=union\n");
    return d;
}

static char *logs_dir(void) {
    char *d = fmtstr("%s/logs", hill_dir());
    mkdir(d, 0777);
    return d;
}

static char *utc_text(int compact) {
    long long t = (long long)time(NULL), days = t / 86400, secs = t % 86400;
    if (secs < 0) secs += 86400, days--;
    long long z = days + 719468, era = (z >= 0 ? z : z - 146096) / 146097;
    long long doe = z - era * 146097, yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    long long doy = doe - (365 * yoe + yoe / 4 - yoe / 100), mp = (5 * doy + 2) / 153;
    long long d = doy - (153 * mp + 2) / 5 + 1, m = mp < 10 ? mp + 3 : mp - 9, y = yoe + era * 400 + (m <= 2);
    if (compact) return fmtstr("%04lld%02lld%02lld-%02lld%02lld%02lld", y, m, d, secs / 3600, secs % 3600 / 60, secs % 60);
    return fmtstr("%04lld-%02lld-%02lldT%02lld:%02lld:%02lldZ", y, m, d, secs / 3600, secs % 3600 / 60, secs % 60);
}

static Strs entries_of(const char *dir) {
    Strs out = {0};
    DIR *d = opendir(dir);
    struct dirent *de;
    while (d && (de = readdir(d)))
        if (strcmp(de->d_name, ".") && strcmp(de->d_name, "..")) spush(&out, strdup(de->d_name));
    if (d) closedir(d);
    return out;
}

static void append(J *entry) {
    J *e = jnew(JOBJ);
    jset(e, "at", jstr(utc_text(0)));
    for (size_t i = 0; i < entry->len; i++) jset(e, entry->k[i], entry->v[i]);
    Buf b = {0};
    dump(&b, e);
    bputc(&b, '\n');
    char *path = fmtstr("%s/ledger.jsonl", hill_dir());
    FILE *f = fopen(path, "a");
    if (!f) stop("ERROR", "cannot write %s", path);
    fwrite(b.p, 1, b.n, f);
    fclose(f);
}

static int lock_fd = -1;

static void lock_hill(void) {
    char *path = fmtstr("%s/lock", hill_dir());
    lock_fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0666);
    if (lock_fd < 0 || flock(lock_fd, LOCK_EX | LOCK_NB)) stop("REFUSE", "another hill run holds the lock; nothing was started");
}

static void unlock_hill(void) {
    if (lock_fd >= 0) close(lock_fd);
    lock_fd = -1;
}

static char *best_tree(Spec *s, const char *commit) {
    char *trees = fmtstr("%s/trees", hill_dir());
    char *path = fmtstr("%s/%.12s", trees, commit);
    char *dotgit = fmtstr("%s/.git", path);
    if (access(dotgit, F_OK)) {
        mkdir(trees, 0777);
        Strs old = entries_of(trees);
        for (size_t i = 0; i < old.n; i++) {
            char *gone = fmtstr("%s/%s", trees, old.v[i]);
            char *argv[] = {"git", "worktree", "remove", "--force", gone, NULL};
            spawn(argv, root_dir, NULL, 0, 1, 1, -1, NULL);
            char *rm[] = {"rm", "-rf", gone, NULL};
            if (!access(gone, F_OK)) spawn(rm, root_dir, NULL, 0, 1, 1, -1, NULL);
        }
        GIT("worktree", "prune");
        GIT("worktree", "add", "--detach", path, commit);
    }
    J *share = jget(s->spec, "share");
    for (size_t i = 0; share && share->t == JARR && i < share->len; i++) {
        char *link = fmtstr("%s/%s", path, share->v[i]->s);
        struct stat st;
        if (lstat(link, &st)) {
            char *parent = strdup(link);
            for (char *q = parent + strlen(path) + 1; (q = strchr(q, '/')); q++) {
                *q = 0;
                mkdir(parent, 0777);
                *q = '/';
            }
            if (symlink(fmtstr("%s/%s", root_dir, share->v[i]->s), link)) stop("ERROR", "cannot link %s", link);
        }
    }
    return path;
}

static Proc shell(const char *cmd, const char *cwd, int capture_out, int capture_err, int out_fd, char *const *env) {
    char *argv[] = {"/bin/sh", "-c", (char *)cmd, NULL};
    return spawn(argv, cwd, NULL, 0, capture_out, capture_err, out_fd, env);
}

static void build(Spec *s, const char *tree, const char *side) {
    const char *cmd = jtext(jget(s->spec, "build"));
    if (!cmd || !cmd[0]) return;
    char *log = fmtstr("%s/%s-build-%s.log", logs_dir(), utc_text(1), side);
    int fd = open(log, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0666);
    Proc r = shell(cmd, tree, 0, 0, fd, NULL);
    close(fd);
    if (r.status) stop("ERROR", "the build failed for the %s (see %s)", side, log);
}

static double available_gb(void) {
#ifdef __APPLE__
    char *argv[] = {"vm_stat", NULL};
    Proc r = spawn(argv, NULL, NULL, 0, 1, 1, -1, NULL);
    double page = 4096;
    char *q = strstr(r.out.p, "page size of ");
    if (q) page = atof(q + 13);
    double pages = 0;
    const char *keys[] = {"Pages free:", "Pages inactive:", "Pages speculative:"};
    for (int i = 0; i < 3; i++)
        if ((q = strstr(r.out.p, keys[i]))) pages += atof(q + strlen(keys[i]));
    return pages * page / 1073741824.0;
#else
    FILE *f = fopen("/proc/meminfo", "r");
    char line[256];
    long long kb = 0;
    while (f && fgets(line, sizeof line, f))
        if (!strncmp(line, "MemAvailable:", 13)) kb = atoll(line + 13);
    if (f) fclose(f);
    return (double)kb / 1048576.0;
#endif
}

static char **env_with(const char *side, const char *tree) {
    size_t n = 0;
    while (environ[n]) n++;
    char **env = malloc((n + 3) * sizeof *env);
    size_t k = 0;
    for (size_t i = 0; i < n; i++)
        if (strncmp(environ[i], "HILL_SIDE=", 10) && strncmp(environ[i], "HILL_TREE=", 10)) env[k++] = environ[i];
    env[k++] = fmtstr("HILL_SIDE=%s", side);
    env[k++] = fmtstr("HILL_TREE=%s", tree);
    env[k] = NULL;
    return env;
}

static int linebreak(unsigned char c) { return c == '\n' || c == '\r' || c == 0x0b || c == 0x0c || c == 0x1c || c == 0x1d || c == 0x1e; }

static J *measure(Spec *s, const char *name, const char *tree, const char *side) {
    J *inst = jget(jget(s->spec, "instruments"), name);
    J *need = jget(inst, "memory_gb");
    if (jnum(need) && jd(need) != 0) {
        double have = available_gb();
        if (have < jd(need)) {
            Buf b = {0};
            dump(&b, need);
            stop("REFUSE", "%s needs %s GB of free memory and %.1f GB is free; nothing was started", name, bstr(&b), available_gb());
        }
    }
    const char *ready = jtext(jget(inst, "ready"));
    if (ready && ready[0]) {
        Proc r = shell(ready, tree, 1, 1, -1, NULL);
        if (r.status) {
            Buf both = {0};
            bputs(&both, r.out.p);
            bputs(&both, r.err.p);
            stop("REFUSE", "%s is not safe to run now: %s", name, strip(bstr(&both)));
        }
    }
    char *log = fmtstr("%s/%s-%s-%s.log", logs_dir(), utc_text(1), name, side);
    int fd = open(log, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0666);
    char *argv[] = {"/bin/sh", "-c", (char *)jtext(jget(inst, "run")), NULL};
    char **env = env_with(side, tree);
    int saved = dup(2);
    dup2(fd, 2);
    Proc r = spawn(argv, tree, NULL, 0, 1, 0, -1, env);
    dup2(saved, 2);
    close(saved);
    close(fd);
    if (r.status) stop("ERROR", "%s failed on the %s with exit %d (see %s)", name, side, r.status, log);
    char *text = r.out.p, *last = NULL;
    size_t lastlen = 0;
    size_t i = 0, n = r.out.n;
    while (i < n) {
        size_t j = i;
        while (j < n && !linebreak((unsigned char)text[j])) j++;
        int blank = 1;
        for (size_t k = i; k < j; k++)
            if (!isspace((unsigned char)text[k])) blank = 0;
        if (!blank) last = text + i, lastlen = j - i;
        if (j < n && text[j] == '\r' && j + 1 < n && text[j + 1] == '\n') j++;
        i = j + 1;
    }
    J *values = last ? parse(last, lastlen) : NULL;
    if (!values || values->t != JOBJ) stop("ERROR", "%s did not end its output with a JSON object (see %s)", name, log);
    return values;
}

static J *field(Spec *s, const char *rid, J *values, const char *name) {
    J *r = jget(jget(s->spec, "rewards"), rid);
    J *v = jget(values, jtext_or(r, "field", rid));
    if (v && v->t == JNULL) return v;
    if (!v || (v->t != JINT && v->t != JREAL) || (v->t == JREAL && !isfinite(v->d))) stop("ERROR", "%s gave no number for %s", name, rid);
    return v;
}

static Strs instrument_names(Spec *s) {
    Strs out = {0};
    J *inst = jget(s->spec, "instruments");
    for (size_t i = 0; i < inst->len; i++) spush(&out, inst->k[i]);
    return out;
}

typedef struct {
    J **v;
    size_t n, cap;
} Vals;

static void vpush(Vals *a, J *v) {
    if (a->n == a->cap) a->cap = a->cap ? 2 * a->cap : 8, a->v = realloc(a->v, a->cap * sizeof *a->v);
    a->v[a->n++] = v;
}

typedef struct {
    Vals *a, *b;
    size_t blocks;
} Sample;

static Strs noisy_instruments(Spec *s) {
    Strs all = instrument_names(s), out = {0};
    J *rewards = jget(s->spec, "rewards");
    for (size_t i = 0; i < all.n; i++)
        for (size_t r = 0; r < rewards->len; r++)
            if (!strcmp(jget(rewards->v[r], "instrument")->s, all.v[i]) && !jtruthy(jget(rewards->v[r], "exact"))) {
                spush(&out, all.v[i]);
                break;
            }
    return out;
}

static int listed(Strs *names, const char *name) {
    for (size_t i = 0; i < names->n; i++)
        if (!strcmp(names->v[i], name)) return 1;
    return 0;
}

static Sample *paired(Spec *s, const char *tree_a, const char *tree_b, const char *a, const char *b, const char *blocks_key, long long dflt,
                      Strs *only) {
    J *rewards = jget(s->spec, "rewards");
    Sample *samples = calloc(rewards->len, sizeof *samples);
    Strs names = only ? *only : instrument_names(s);
    for (size_t ni = 0; ni < names.n; ni++) {
        const char *name = names.v[ni];
        long long n = jint_or(jget(jget(s->spec, "instruments"), name), blocks_key, dflt);
        for (size_t r = 0; r < rewards->len; r++)
            if (!strcmp(jget(rewards->v[r], "instrument")->s, name)) {
                samples[r].a = calloc((size_t)n, sizeof(Vals));
                samples[r].b = calloc((size_t)n, sizeof(Vals));
                samples[r].blocks = (size_t)n;
            }
        for (long long k = 0; k < n; k++) {
            const char *order[4] = {a, b, b, a};
            if (k % 2) order[0] = b, order[1] = a, order[2] = a, order[3] = b;
            printf("hill: %s, block %lld of %lld: %s, %s, %s, %s\n", name, k + 1, n, order[0], order[1], order[2], order[3]);
            fflush(stdout);
            J *runs[4];
            for (int i = 0; i < 4; i++) runs[i] = measure(s, name, order[i] == a ? tree_a : tree_b, order[i]);
            for (size_t r = 0; r < rewards->len; r++) {
                if (strcmp(jget(rewards->v[r], "instrument")->s, name)) continue;
                for (int i = 0; i < 4; i++)
                    if (order[i] == a) vpush(&samples[r].a[k], field(s, rewards->k[r], runs[i], name));
                for (int i = 0; i < 4; i++)
                    if (order[i] == b) vpush(&samples[r].b[k], field(s, rewards->k[r], runs[i], name));
            }
        }
    }
    return samples;
}

static Vals *solo(Spec *s, const char *tree, const char *side) {
    J *rewards = jget(s->spec, "rewards");
    Vals *samples = calloc(rewards->len, sizeof *samples);
    Strs names = instrument_names(s);
    for (size_t ni = 0; ni < names.n; ni++) {
        const char *name = names.v[ni];
        long long n = 2 * jint_or(jget(jget(s->spec, "instruments"), name), "blocks", BLOCKS);
        for (long long k = 0; k < n; k++) {
            printf("hill: %s, run %lld of %lld\n", name, k + 1, n);
            fflush(stdout);
            J *values = measure(s, name, tree, side);
            for (size_t r = 0; r < rewards->len; r++)
                if (!strcmp(jget(rewards->v[r], "instrument")->s, name)) vpush(&samples[r], field(s, rewards->k[r], values, name));
        }
    }
    return samples;
}

static Vals flat(Vals *blocks, size_t n) {
    Vals out = {0};
    for (size_t k = 0; k < n; k++)
        for (size_t i = 0; i < blocks[k].n; i++) vpush(&out, blocks[k].v[i]);
    return out;
}

static double fsum(const double *xs, size_t count) {
    size_t cap = 32, n = 0;
    double *p = malloc(cap * sizeof *p);
    double special = 0, infs = 0;
    for (size_t k = 0; k < count; k++) {
        double x = xs[k], xsave = x, y, t, hi, yr, lo = 0;
        size_t i = 0;
        for (size_t j = 0; j < n; j++) {
            y = p[j];
            if (fabs(x) < fabs(y)) t = x, x = y, y = t;
            hi = x + y;
            yr = hi - x;
            lo = y - yr;
            if (lo != 0.0) p[i++] = lo;
            x = hi;
        }
        n = i;
        if (x != 0.0) {
            if (!isfinite(x)) {
                if (isinf(xsave)) infs += xsave;
                special += xsave;
                n = 0;
            } else {
                if (n >= cap) cap *= 2, p = realloc(p, cap * sizeof *p);
                p[n++] = x;
            }
        }
    }
    if (special != 0.0) {
        free(p);
        return isnan(infs) ? NAN : special;
    }
    double hi = 0.0, lo = 0.0, x, y, yr;
    if (n > 0) {
        hi = p[--n];
        while (n > 0) {
            x = hi;
            y = p[--n];
            hi = x + y;
            yr = hi - x;
            lo = y - yr;
            if (lo != 0.0) break;
        }
        if (n > 0 && ((lo < 0.0 && p[n - 1] < 0.0) || (lo > 0.0 && p[n - 1] > 0.0))) {
            y = lo * 2.0;
            x = hi + y;
            yr = x - hi;
            if (y == yr) hi = x;
        }
    }
    free(p);
    return hi;
}

static double fmean_vals(Vals *v) {
    double *xs = malloc((v->n + 1) * sizeof *xs);
    for (size_t i = 0; i < v->n; i++) xs[i] = jd(v->v[i]);
    double m = fsum(xs, v->n) / (double)v->n;
    free(xs);
    return m;
}

static double fmean(const double *xs, size_t n) { return fsum(xs, n) / (double)n; }

static int num_cmp(const void *a, const void *b) {
    double x = jd(*(J *const *)a), y = jd(*(J *const *)b);
    return (x > y) - (x < y);
}

static J *median(Vals *v) {
    J **sorted = malloc(v->n * sizeof *sorted);
    memcpy(sorted, v->v, v->n * sizeof *sorted);
    for (size_t i = 1; i < v->n; i++) {
        J *key = sorted[i];
        size_t k = i;
        while (k > 0 && num_cmp(&sorted[k - 1], &key) > 0) sorted[k] = sorted[k - 1], k--;
        sorted[k] = key;
    }
    if (v->n % 2 == 1) return sorted[v->n / 2];
    J *x = sorted[v->n / 2 - 1], *y = sorted[v->n / 2];
    if (x->t == JINT && y->t == JINT) return jreal((double)(x->i + y->i) / 2);
    return jreal((jd(x) + jd(y)) / 2);
}

static double stdev(const double *xs, size_t n) {
    long double mean = 0, ss = 0;
    for (size_t i = 0; i < n; i++) mean += xs[i];
    mean /= (long double)n;
    for (size_t i = 0; i < n; i++) ss += ((long double)xs[i] - mean) * ((long double)xs[i] - mean);
    return (double)sqrtl(ss / (long double)(n - 1));
}

static double inv_cdf(double p) {
    double q = p - 0.5, r, num, den, x;
    if (fabs(q) <= 0.425) {
        r = 0.180625 - q * q;
        num = (((((((2.5090809287301226727e+3 * r + 3.3430575583588128105e+4) * r + 6.7265770927008700853e+4) * r +
                   4.5921953931549871457e+4) * r + 1.3731693765509461125e+4) * r + 1.9715909503065514427e+3) * r +
                1.3314166789178437745e+2) * r + 3.3871328727963666080e+0) * q;
        den = (((((((5.2264952788528545610e+3 * r + 2.8729085735721942674e+4) * r + 3.9307895800092710610e+4) * r +
                   2.1213794301586595867e+4) * r + 5.3941960214247511077e+3) * r + 6.8718700749205790830e+2) * r +
                4.2313330701600911252e+1) * r + 1.0);
        x = num / den;
        return 0.0 + (x * 1.0);
    }
    r = q <= 0.0 ? p : 1.0 - p;
    r = sqrt(-log(r));
    if (r <= 5.0) {
        r = r - 1.6;
        num = (((((((7.7454501427834140764e-4 * r + 2.2723844989269184583e-2) * r + 2.4178072517745061177e-1) * r +
                   1.2704582524523683826e+0) * r + 3.6478483247632045960e+0) * r + 5.7694972214606914055e+0) * r +
                4.6303378461565452959e+0) * r + 1.4234371107496835773e+0);
        den = (((((((1.0507500716444168432e-9 * r + 5.4759380849953449460e-4) * r + 1.5198666563616457197e-2) * r +
                   1.4810397642748007459e-1) * r + 6.8976733498510000455e-1) * r + 1.6763848301838038494e+0) * r +
                2.0531916266377588219e+0) * r + 1.0);
    } else {
        r = r - 5.0;
        num = (((((((2.0103343992922881327e-7 * r + 2.7115555687434875782e-5) * r + 1.2426609473880784386e-3) * r +
                   2.6532189526576123093e-2) * r + 2.9656057182850489123e-1) * r + 1.7848265399172913358e+0) * r +
                5.4637849111641143699e+0) * r + 6.6579046435011037772e+0);
        den = (((((((2.0442631033899397856e-15 * r + 1.4215117583164458887e-7) * r + 1.8463183175100546818e-5) * r +
                   7.8686913114561329059e-4) * r + 1.4875361290850614853e-2) * r + 1.3692988092273580531e-1) * r +
                5.9983220655588793769e-1) * r + 1.0);
    }
    x = num / den;
    if (q < 0.0) x = -x;
    return 0.0 + (x * 1.0);
}

static double quantile(double p, long long dof) {
    double z = inv_cdf(p);
    return z + (pow(z, 3.0) + z) / (double)(4 * dof) + (5 * pow(z, 5.0) + 16 * pow(z, 3.0) + 3 * z) / (double)(96 * dof * dof);
}

typedef struct {
    double sigma;
    long long dof;
    int has, na;
} Noise;

static double floor_of(Spec *s, Noise noise, size_t n) {
    J *rewards = jget(s->spec, "rewards");
    long long noisy = 0;
    for (size_t i = 0; i < rewards->len; i++) noisy += !jtruthy(jget(rewards->v[i], "exact"));
    double m = (double)(noisy > 1 ? noisy : 1);
    return quantile(1 - ALPHA / m, noise.dof) * noise.sigma / sqrt((double)n);
}

static Noise *calibration(Spec *s, Ledger *l, const char *ident, const char *harness) {
    J *rewards = jget(s->spec, "rewards");
    Noise *out = calloc(rewards->len, sizeof *out);
    for (size_t r = 0; r < rewards->len; r++) {
        size_t count = 0, cap = 16;
        int absent = 0;
        double *xs = malloc(cap * sizeof *xs);
        for (size_t i = 0; i < l->n; i++) {
            J *e = l->v[i];
            if (!is(e, "type", "calibration") || !is(e, "machine", ident) || !is(e, "harness", harness)) continue;
            J *xsj = jget(jget(e, "blocks"), rewards->k[r]);
            absent |= xsj && xsj->t == JNULL;
            for (size_t k = 0; xsj && k < xsj->len; k++) {
                if (count == cap) cap *= 2, xs = realloc(xs, cap * sizeof *xs);
                double x = jd(xsj->v[k]);
                xs[count++] = x * x;
            }
        }
        if (count >= 3) out[r] = (Noise){sqrt(fmean(xs, count)), (long long)count, 1, 0};
        else out[r].na = absent;
        free(xs);
    }
    return out;
}

static int unit_is(const char *unit, const char *x) { return unit && !strcmp(unit, x); }

static void commas(Buf *b, const char *digits) {
    const char *d = digits;
    if (*d == '-') bputc(b, '-'), d++;
    size_t n = strlen(d);
    for (size_t i = 0; i < n; i++) {
        if (i && (n - i) % 3 == 0) bputc(b, ',');
        bputc(b, d[i]);
    }
}

static char *fmt_value(Spec *s, const char *rid, J *v, int digits) {
    if (!v || v->t == JNULL) return "—";
    J *r = jget(jget(s->spec, "rewards"), rid);
    const char *unit = jtext_or(r, "unit", "");
    int d = digits ? digits : (int)jint_or(r, "digits", 3);
    double x = jd(v);
    Buf b = {0};
    if (!strcmp(unit, "%")) {
        bprintf(&b, "%.*f%%", d - 2 > 0 ? d - 2 : 0, 100 * x);
        return bstr(&b);
    }
    Buf text = {0};
    if (v->t == JINT && (v->i >= 1 || v->i <= -1)) {
        char tmp[32];
        snprintf(tmp, sizeof tmp, "%lld", v->i);
        commas(&text, tmp);
    } else if (v->t == JREAL && x == floor(x) && fabs(x) >= 1) {
        char *tmp = fmtstr("%.0f", x);
        commas(&text, tmp);
    } else if (fabs(x) >= pow(10, d - 1)) {
        char *tmp = fmtstr("%.0f", x);
        commas(&text, tmp);
    } else if (x == 0) {
        bputs(&text, "0");
    } else {
        int places = d - 1 - (int)floor(log10(fabs(x)));
        bprintf(&text, "%.*f", places > 0 ? places : 0, x);
    }
    if (unit_is(unit, "×") || unit_is(unit, "x")) {
        bputs(&b, bstr(&text));
        bputs(&b, unit);
        return bstr(&b);
    }
    bprintf(&b, "%s %s", bstr(&text), unit);
    return strip(bstr(&b));
}

static int num_eq(J *a, J *b) {
    if (a->t == JNULL || b->t == JNULL) return a->t == b->t;
    if (a->t == JINT && b->t == JINT) return a->i == b->i;
    return jd(a) == jd(b);
}

static void pair_fmt(Spec *s, const char *rid, J *a, J *c, char **fa, char **fc) {
    J *r = jget(jget(s->spec, "rewards"), rid);
    int d = (int)jint_or(r, "digits", 3);
    while (!num_eq(a, c) && !strcmp(fmt_value(s, rid, a, d), fmt_value(s, rid, c, d)) && d < 17) d++;
    *fa = fmt_value(s, rid, a, d);
    *fc = fmt_value(s, rid, c, d);
}

static char *name_of(Spec *s, const char *rid) {
    J *r = jget(jget(s->spec, "rewards"), rid);
    Buf b = {0};
    const char *parts[3] = {jtext_or(r, "label", rid), jtext(jget(r, "case")), jtext(jget(r, "config"))};
    int first = 1;
    for (int i = 0; i < 3; i++)
        if (parts[i] && parts[i][0]) {
            if (!first) bputs(&b, ", ");
            bputs(&b, parts[i]);
            first = 0;
        }
    return bstr(&b);
}

static int nulls(const char *rid, Vals *v) {
    size_t none = 0;
    for (size_t i = 0; i < v->n; i++) none += v->v[i]->t == JNULL;
    if (none && none < v->n) stop("ERROR", "%s gave a number on some runs and null on others", rid);
    return none == v->n;
}

static J *exact_value(const char *rid, Vals *v) {
    if (nulls(rid, v)) return jnew(JNULL);
    for (size_t i = 1; i < v->n; i++)
        if (!num_eq(v->v[i], v->v[0])) {
            Vals distinct = {0};
            for (size_t k = 0; k < v->n; k++) {
                int seen = 0;
                for (size_t q = 0; q < distinct.n; q++) seen |= num_eq(distinct.v[q], v->v[k]);
                if (!seen) vpush(&distinct, v->v[k]);
            }
            qsort(distinct.v, distinct.n, sizeof *distinct.v, num_cmp);
            Buf b = {0};
            bputc(&b, '[');
            for (size_t k = 0; k < distinct.n; k++) {
                if (k) bputs(&b, ", ");
                dump(&b, distinct.v[k]);
            }
            bputc(&b, ']');
            stop("ERROR", "%s is marked exact but gave %s for the same code; it is noisy", rid, bstr(&b));
        }
    return v->v[0];
}

static int exact_only(Spec *s, const char *name) {
    J *rewards = jget(s->spec, "rewards");
    for (size_t r = 0; r < rewards->len; r++)
        if (!strcmp(jget(rewards->v[r], "instrument")->s, name) && !jtruthy(jget(rewards->v[r], "exact"))) return 0;
    return 1;
}

static J *known_best(Spec *s, Ledger *l, const char *machine, const char *harness, const char *product, const char *name) {
    J *rewards = jget(s->spec, "rewards"), *found = NULL;
    for (size_t i = 0; i < l->n; i++) {
        J *e = l->v[i], *vals = jnew(JOBJ);
        if (!is(e, "machine", machine) || !is(e, "harness", harness)) continue;
        int all = 1;
        for (size_t r = 0; r < rewards->len && all; r++) {
            if (strcmp(jget(rewards->v[r], "instrument")->s, name)) continue;
            J *v = NULL;
            if (is(e, "type", "record") && is(e, "product", product)) v = jget(jget(e, "values"), rewards->k[r]);
            else if (is(e, "type", "verdict") && is(e, "candidate", product)) v = jget(jget(e, "values"), rewards->k[r]);
            else if (is(e, "type", "verdict") && is(e, "best", product)) v = jget(jget(jget(e, "rows"), rewards->k[r]), "best");
            if (v) jset(vals, rewards->k[r], v);
            else all = 0;
        }
        if (all) found = vals;
    }
    return found;
}

static Sample *once(Spec *s, const char *tree_a, const char *tree_b, const char *name, J *known) {
    J *rewards = jget(s->spec, "rewards");
    Sample *samples = calloc(rewards->len, sizeof *samples);
    printf(known ? "hill: %s: candidate, against the best's values in the ledger\n" : "hill: %s: best, candidate\n", name);
    fflush(stdout);
    J *best = known ? known : measure(s, name, tree_a, "best"), *cand = measure(s, name, tree_b, "candidate");
    int moved = 0;
    for (size_t r = 0; r < rewards->len; r++) {
        if (strcmp(jget(rewards->v[r], "instrument")->s, name)) continue;
        samples[r].a = calloc(1, sizeof(Vals));
        samples[r].b = calloc(1, sizeof(Vals));
        samples[r].blocks = 1;
        vpush(&samples[r].a[0], known ? jget(known, rewards->k[r]) : field(s, rewards->k[r], best, name));
        vpush(&samples[r].b[0], field(s, rewards->k[r], cand, name));
        moved |= !num_eq(samples[r].a[0].v[0], samples[r].b[0].v[0]);
    }
    if (moved && known) {
        printf("hill: %s: best, to confirm the ledger\n", name);
        fflush(stdout);
        J *fresh = measure(s, name, tree_a, "best");
        for (size_t r = 0; r < rewards->len; r++)
            if (samples[r].blocks) vpush(&samples[r].a[0], field(s, rewards->k[r], fresh, name));
    }
    if (moved) {
        printf("hill: %s: candidate again\n", name);
        fflush(stdout);
        J *again = measure(s, name, tree_b, "candidate");
        for (size_t r = 0; r < rewards->len; r++)
            if (samples[r].blocks) vpush(&samples[r].b[0], field(s, rewards->k[r], again, name));
    }
    return samples;
}

typedef struct {
    J *rows;
    int wins, losses;
} Judged;

static Judged judge(Spec *s, Sample *samples, Noise *noise) {
    J *rewards = jget(s->spec, "rewards");
    Judged out = {jnew(JOBJ), 0, 0};
    for (size_t r = 0; r < rewards->len; r++) {
        if (!samples[r].blocks) continue;
        const char *rid = rewards->k[r];
        J *spec = rewards->v[r];
        double sign = is(spec, "better", "higher") ? 1 : -1;
        J *row = jnew(JOBJ);
        const char *result;
        Sample *sm = &samples[r];
        Vals fa = flat(sm->a, sm->blocks), fc = flat(sm->b, sm->blocks);
        int exact = jtruthy(jget(spec, "exact")), gone = nulls(rid, &fa), missing = nulls(rid, &fc);
        if (gone || missing) {
            jset(row, "best", gone ? jnew(JNULL) : exact ? exact_value(rid, &fa) : median(&fa));
            jset(row, "candidate", missing ? jnew(JNULL) : exact ? exact_value(rid, &fc) : median(&fc));
            jset(row, "change", jnew(JNULL));
            result = gone && missing ? "tie" : gone ? "win" : "loss";
        } else if (exact) {
            J *a = exact_value(rid, &fa), *c = exact_value(rid, &fc);
            jset(row, "best", a);
            jset(row, "candidate", c);
            int zero = a->t == JINT ? a->i == 0 : a->d == 0;
            jset(row, "change", zero ? jnew(JNULL) : jreal(jd(c) / jd(a) - 1));
            double diff = (a->t == JINT && c->t == JINT) ? (double)(c->i - a->i) : jd(c) - jd(a);
            result = diff * sign > 0 ? "win" : diff * sign < 0 ? "loss" : "tie";
        } else {
            double *xs = malloc((sm->blocks + 1) * sizeof *xs);
            for (size_t k = 0; k < sm->blocks; k++) {
                double ma = fmean_vals(&sm->a[k]), mc = fmean_vals(&sm->b[k]);
                if (ma <= 0 || mc <= 0) stop("ERROR", "%s is noisy but not positive; it has to be exact", rid);
                xs[k] = log(mc / ma);
            }
            double d = fmean(xs, sm->blocks);
            Noise nz = noise[r];
            if (nz.na) stop("ERROR", "%s now gives numbers but was calibrated as not applicable here: run hill calibrate", rid);
            if (!nz.has) {
                double *reps = malloc(2 * sm->blocks * sizeof *reps);
                size_t n = 0;
                for (int side = 0; side < 2; side++)
                    for (size_t k = 0; k < sm->blocks; k++, n++) {
                        Vals *v = side ? &sm->b[k] : &sm->a[k];
                        reps[n] = log(jd(v->v[v->n - 1]) / jd(v->v[0]));
                        reps[n] *= reps[n];
                    }
                nz = (Noise){sqrt(fsum(reps, n) / (double)n / 2), (long long)n, 1, 0};
                free(reps);
            }
            double fl = floor_of(s, nz, sm->blocks);
            jset(row, "best", median(&fa));
            jset(row, "candidate", median(&fc));
            jset(row, "change", jreal(expm1(d)));
            jset(row, "floor", jreal(expm1(fl)));
            result = d * sign > fl ? "win" : d * sign < -fl ? "loss" : "tie";
            free(xs);
        }
        jset(row, "result", jstr(result));
        jset(out.rows, rid, row);
        out.wins += !strcmp(result, "win");
        out.losses += !strcmp(result, "loss");
    }
    return out;
}

static char *report(Spec *s, J *rows, const char *verdict) {
    Buf b = {0}, named = {0};
    bputs(&b, "| Reward | Best | Candidate | Change | Verdict |\n| --- | ---: | ---: | ---: | --- |");
    const char *pick = !strcmp(verdict, "PROMOTE") ? "win" : !strcmp(verdict, "REJECT") ? "loss" : "";
    int counts[3] = {0};
    for (size_t i = 0; i < rows->len; i++) {
        J *row = rows->v[i];
        J *change = jget(row, "change");
        Buf c = {0}, d = {0};
        if (!change || change->t == JNULL) {
        } else if (jget(row, "floor")) {
            bprintf(&d, "%+.2f%%", 100 * jd(change));
            bprintf(&c, "%s (floor \xc2\xb1%.2f%%)", bstr(&d), 100 * jd(jget(row, "floor")));
        } else {
            bprintf(&d, "%+.3g%%", 100 * jd(change));
            bputs(&c, bstr(&d));
        }
        char *fa, *fc;
        pair_fmt(s, rows->k[i], jget(row, "best"), jget(row, "candidate"), &fa, &fc);
        const char *result = jget(row, "result")->s;
        bprintf(&b, "\n| %s | %s | %s | %s | %s |", name_of(s, rows->k[i]), fa, fc, bstr(&c), result);
        if (!strcmp(result, pick)) bprintf(&named, "\n  %s: %s \xe2\x86\x92 %s%s%s%s", name_of(s, rows->k[i]), fa, fc, d.n ? " (" : "", bstr(&d), d.n ? ")" : "");
        counts[!strcmp(result, "win") ? 0 : !strcmp(result, "tie") ? 1 : 2]++;
    }
    size_t total = jget(s->spec, "rewards")->len;
    if (rows->len < total) bprintf(&b, "\n\nThe gate stopped at the first loss: %zu of %zu rewards were not measured.", total - rows->len, total);
    if (!strcmp(verdict, "PROMOTE"))
        bprintf(&b, "\n\n\xf0\x9f\x91\x91 PROMOTE on %d reward%s, with %d equal and none worse:%s\nCommit it with .hill/ledger.jsonl staged; it's the new best.", counts[0],
                counts[0] == 1 ? "" : "s", counts[1], bstr(&named));
    else if (!strcmp(verdict, "REJECT"))
        bprintf(&b, "\n\nREJECT on %d reward%s, with %d better and %d equal:%s\nThe best stays.", counts[2], counts[2] == 1 ? "" : "s", counts[0], counts[1], bstr(&named));
    else
        bprintf(&b, "\n\nTIE: %d better, %d equal, %d worse. The best stays.", counts[0], counts[1], counts[2]);
    return bstr(&b);
}

typedef struct {
    char *ident, *label;
} Machine;

static int path_parts_cmp(const void *a, const void *b) {
    const char *x = *(const char *const *)a, *y = *(const char *const *)b;
    while (*x && *y) {
        size_t nx = strcspn(x, "/"), ny = strcspn(y, "/");
        size_t m = nx < ny ? nx : ny;
        int c = strncmp(x, y, m);
        if (c) return c;
        if (nx != ny) return nx < ny ? -1 : 1;
        x += nx, y += ny;
        if (*x == '/') x++;
        if (*y == '/') y++;
    }
    return (*x != 0) - (*y != 0);
}

static char *on_path(const char *program) {
    const char *path = getenv("PATH");
    if (!path) return NULL;
    char *copy = strdup(path), *save = NULL;
    for (char *dir = strtok_r(copy, ":", &save); dir; dir = strtok_r(NULL, ":", &save)) {
        char *full = fmtstr("%s/%s", dir, program);
        if (!access(full, X_OK)) return full;
    }
    return NULL;
}

static Strs shlex(const char *s) {
    Strs out = {0};
    while (*s) {
        while (*s && isspace((unsigned char)*s)) s++;
        if (!*s) break;
        Buf w = {0};
        while (*s && !isspace((unsigned char)*s)) {
            if (*s == '"') {
                s++;
                while (*s && *s != '"') {
                    if (*s == '\\' && (s[1] == '"' || s[1] == '\\' || s[1] == '$' || s[1] == '`')) s++;
                    bputc(&w, *s++);
                }
                if (*s) s++;
            } else if (*s == '\'') {
                s++;
                while (*s && *s != '\'') bputc(&w, *s++);
                if (*s) s++;
            } else if (*s == '\\' && s[1]) {
                bputc(&w, s[1]);
                s += 2;
            } else
                bputc(&w, *s++);
        }
        spush(&out, bstr(&w));
    }
    return out;
}

static Machine machine(Spec *s) {
    J *info = jnew(JOBJ);
    char *label;
#ifdef __APPLE__
    char *argv1[] = {"sysctl", "-n", "machdep.cpu.brand_string", NULL};
    char *argv2[] = {"sysctl", "-n", "hw.memsize", NULL};
    Proc a = spawn(argv1, NULL, NULL, 0, 1, 1, -1, NULL), m = spawn(argv2, NULL, NULL, 0, 1, 1, -1, NULL);
    J *mem = jnew(JINT);
    mem->i = atoll(strip(m.out.p)) >> 30;
    jset(info, "cpu", jstr(strip(a.out.p)));
    jset(info, "memory_gb", mem);
    label = fmtstr("%s, %lld GB", strip(a.out.p), mem->i);
#else
    char *cpu = NULL, line[1024];
    FILE *f = fopen("/proc/cpuinfo", "r");
    while (f && fgets(line, sizeof line, f))
        if (!cpu && !strncmp(line, "model name", 10) && strchr(line, ':')) cpu = strdup(strip(strchr(line, ':') + 1));
    if (f) fclose(f);
    if (!cpu) {
        struct utsname u;
        uname(&u);
        cpu = strdup(u.machine);
    }
    long long kb = 0;
    f = fopen("/proc/meminfo", "r");
    while (f && fgets(line, sizeof line, f))
        if (!strncmp(line, "MemTotal:", 9)) kb = atoll(line + 9);
    if (f) fclose(f);
    J *mem = jnew(JINT);
    mem->i = kb >> 20;
    J *cards = jnew(JARR);
    Strs names = entries_of("/sys/class/drm"), devices = {0};
    for (size_t i = 0; i < names.n; i++) {
        struct stat st;
        char *device = fmtstr("/sys/class/drm/%s/device", names.v[i]);
        if (!strncmp(names.v[i], "card", 4) && isdigit((unsigned char)names.v[i][4]) && !lstat(device, &st)) spush(&devices, device);
    }
    if (devices.n) {
        qsort(devices.v, devices.n, sizeof *devices.v, path_parts_cmp);
        char *lspci = on_path("lspci");
        for (size_t i = 0; i < devices.n; i++) {
            char *vf = fmtstr("%s/mem_info_vram_total", devices.v[i]);
            char *text = read_file(vf, NULL);
            long long vram = 0;
            if (text) {
                char *end;
                char *t = strip(text);
                vram = strtoll(t, &end, 10);
                if (*end || !*t) vram = 0;
            }
            char *real = realpath(devices.v[i], NULL);
            char *name = strdup(real ? strrchr(real, '/') + 1 : devices.v[i]);
            char *address = strdup(name);
            if (lspci) {
                char *argv[] = {lspci, "-mm", "-s", address, NULL};
                Proc r = spawn(argv, NULL, NULL, 0, 1, 1, -1, NULL);
                Strs fields = shlex(r.out.n ? r.out.p : "\"\"");
                if (fields.n > 3) name = fields.v[3];
            }
            J *card = jnew(JOBJ);
            J *gb = jnew(JINT);
            gb->i = vram >> 30;
            jset(card, "name", jstr(name));
            jset(card, "vram_gb", gb);
            jpush(cards, card);
        }
    }
    for (size_t i = 1; i < cards->len; i++) {
        J *key = cards->v[i];
        size_t k = i;
        while (k > 0 && jget(cards->v[k - 1], "vram_gb")->i < jget(key, "vram_gb")->i) cards->v[k] = cards->v[k - 1], k--;
        cards->v[k] = key;
    }
    jset(info, "cpu", jstr(cpu));
    jset(info, "memory_gb", mem);
    jset(info, "gpus", cards);
    label = cards->len && jget(cards->v[0], "vram_gb")->i >= 2 ? jget(cards->v[0], "name")->s : cpu;
#endif
    struct utsname u;
    uname(&u);
    jset(info, "os", jstr(u.sysname));
    jset(info, "arch", jstr(u.machine));
    Machine out = {digest_json(info), label};
    const char *named = jtext(jget(jget(s->spec, "machines"), out.ident));
    if (named) out.label = (char *)named;
    return out;
}

static J *entry_of(const char *type, Machine *m) {
    J *e = jnew(JOBJ);
    jset(e, "type", jstr(type));
    jset(e, "machine", jstr(m->ident));
    jset(e, "label", jstr(m->label));
    return e;
}

static int cmd_calibrate(Spec *s) {
    char *commit = head_commit();
    if (!commit) stop("REFUSE", "commit something first: calibration measures HEAD");
    Digests best = digests(s, commit), work = digests(s, "worktree");
    if (strcmp(best.product, work.product) || strcmp(best.harness, work.harness))
        stop("REFUSE", "calibration pairs HEAD's tree with the working tree, so the working tree's measured and harness paths have to match HEAD's");
    Machine m = machine(s);
    lock_hill();
    char *tree = best_tree(s, commit);
    build(s, tree, "best");
    build(s, root_dir, "again");
    Strs names = noisy_instruments(s);
    Sample *samples = paired(s, tree, root_dir, "best", "again", "calibration_blocks", CALIBRATION_BLOCKS, &names);
    unlock_hill();
    J *rewards = jget(s->spec, "rewards");
    J *blocks = jnew(JOBJ);
    Buf biased = {0};
    int nbiased = 0;
    for (size_t r = 0; r < rewards->len; r++) {
        const char *rid = rewards->k[r];
        Sample *sm = &samples[r];
        if (!listed(&names, jget(rewards->v[r], "instrument")->s)) continue;
        Vals all = flat(sm->a, sm->blocks), again = flat(sm->b, sm->blocks);
        for (size_t i = 0; i < again.n; i++) vpush(&all, again.v[i]);
        if (jtruthy(jget(rewards->v[r], "exact"))) {
            exact_value(rid, &all);
            continue;
        }
        if (nulls(rid, &all)) {
            jset(blocks, rid, jnew(JNULL));
            continue;
        }
        double *xs = malloc((sm->blocks + 1) * sizeof *xs);
        J *list = jnew(JARR);
        for (size_t k = 0; k < sm->blocks; k++) {
            xs[k] = log(fmean_vals(&sm->b[k]) / fmean_vals(&sm->a[k]));
            jpush(list, jreal(xs[k]));
        }
        if (sm->blocks < 3) stop("ERROR", "calibrating %s needs at least 3 blocks", rid);
        double sd = stdev(xs, sm->blocks);
        if (sd != 0 && fabs(fmean(xs, sm->blocks)) > 3 * sd / sqrt((double)sm->blocks)) {
            if (nbiased++) bputs(&biased, "\n  ");
            bprintf(&biased, "%s: %+.2f%%", name_of(s, rid), 100 * expm1(fmean(xs, sm->blocks)));
        }
        jset(blocks, rid, list);
    }
    if (nbiased)
        stop("ERROR", "the same code measures differently in HEAD's tree and the working tree, so the gate can't be trusted until "
                      "that's found and fixed (shared build outputs, paths baked into builds, caches):\n  %s", bstr(&biased));
    J *entry = entry_of("calibration", &m);
    jset(entry, "harness", jstr(best.harness));
    jset(entry, "commit", jstr(commit));
    jset(entry, "blocks", blocks);
    append(entry);
    Ledger l = read_ledger();
    Noise *noise = calibration(s, &l, m.ident, best.harness);
    for (size_t r = 0; r < rewards->len; r++) {
        J *b = jget(blocks, rewards->k[r]);
        if (!b || b->t == JNULL) continue;
        long long n = jint_or(jget(jget(s->spec, "instruments"), jget(rewards->v[r], "instrument")->s), "blocks", BLOCKS);
        printf("%s: a gate of %lld blocks sees a change beyond \xc2\xb1%.2f%% (%lld calibration blocks)\n", name_of(s, rewards->k[r]), n,
               100 * expm1(floor_of(s, noise[r], (size_t)n)), noise[r].dof);
    }
    return 0;
}

static int cmd_record(Spec *s) {
    char *commit = head_commit();
    if (!commit) stop("REFUSE", "commit something first: record measures HEAD");
    Digests best = digests(s, commit);
    Machine m = machine(s);
    lock_hill();
    char *tree = best_tree(s, commit);
    build(s, tree, "best");
    Vals *samples = solo(s, tree, "best");
    unlock_hill();
    J *rewards = jget(s->spec, "rewards");
    J *values = jnew(JOBJ);
    for (size_t r = 0; r < rewards->len; r++)
        jset(values, rewards->k[r],
             jtruthy(jget(rewards->v[r], "exact")) || nulls(rewards->k[r], &samples[r]) ? exact_value(rewards->k[r], &samples[r]) : median(&samples[r]));
    Ledger l = read_ledger();
    J *before = NULL;
    for (size_t i = 0; i < l.n; i++) {
        J *e = l.v[i];
        if (is(e, "type", "record") && is(e, "machine", m.ident) && is(e, "product", best.product) && !is(e, "harness", best.harness)) before = e;
    }
    J *entry = entry_of("record", &m);
    jset(entry, "commit", jstr(commit));
    jset(entry, "product", jstr(best.product));
    jset(entry, "harness", jstr(best.harness));
    jset(entry, "values", values);
    append(entry);
    for (size_t r = 0; r < values->len; r++) printf("%s: %s\n", name_of(s, values->k[r]), fmt_value(s, values->k[r], values->v[r], 0));
    if (before) {
        J *old = jget(before, "values");
        Noise *now_noise = calibration(s, &l, m.ident, best.harness), *old_noise = calibration(s, &l, m.ident, jtext(jget(before, "harness")));
        Buf lower = {0};
        int nl = 0;
        for (size_t r = 0; r < values->len; r++) {
            J *o = jget(old, values->k[r]);
            if (!o || o->t == JNULL || values->v[r]->t == JNULL) continue;
            J *spec_r = rewards->v[r];
            double sign = is(spec_r, "better", "higher") ? 1 : -1, v = jd(values->v[r]), was = jd(o);
            Noise noise = now_noise[r].has ? now_noise[r] : old_noise[r];
            for (size_t i = l.n; !noise.has && i-- > 0;) {
                J *e = l.v[i];
                if (is(e, "type", "calibration") && is(e, "machine", m.ident) && jget(jget(e, "blocks"), values->k[r]))
                    noise = calibration(s, &l, m.ident, jtext(jget(e, "harness")))[r];
            }
            int worse;
            if (jtruthy(jget(spec_r, "exact")) || !noise.has || v <= 0 || was <= 0)
                worse = (v - was) * sign < 0;
            else
                worse = log(v / was) * sign <
                        -floor_of(s, noise, (size_t)jint_or(jget(jget(s->spec, "instruments"), jget(spec_r, "instrument")->s), "blocks", BLOCKS));
            if (worse) {
                nl++;
                bprintf(&lower, "  %s: %s -> %s\n", name_of(s, values->k[r]), fmt_value(s, values->k[r], o, 0),
                        fmt_value(s, values->k[r], values->v[r], 0));
            }
        }
        if (nl) {
            printf("\nThe harness changed since the last record of this code, and these values moved the wrong way. "
                   "Whether to keep that harness change is the user's call:\n%s", bstr(&lower));
        }
    }
    return 0;
}

static int cmd_gate(Spec *s) {
    char *commit = head_commit();
    if (!commit) stop("REFUSE", "commit something first: the gate judges the working tree against HEAD");
    Digests best = digests(s, commit);
    Pairs cand_files = files_of(s);
    Digests cand = all_plain(s) ? split_digests(s, &cand_files) : digests(s, "worktree");
    if (strcmp(cand.harness, best.harness)) stop("REFUSE", "the harness differs from HEAD's: land harness changes on their own, then record");
    if (!strcmp(cand.product, best.product)) stop("REFUSE", "the measured paths are the same as HEAD's: nothing to judge");
    Machine m = machine(s);
    Ledger l = read_ledger();
    for (size_t i = 0; i < l.n; i++) {
        J *e = l.v[i];
        if (is(e, "type", "verdict") && is(e, "machine", m.ident) && is(e, "best", best.product) && is(e, "candidate", cand.product) &&
            is(e, "harness", best.harness))
            stop("REFUSE", "this candidate already has a verdict on this machine (%s); a new verdict needs a new candidate", jtext(jget(e, "verdict")));
    }
    Noise *noise = calibration(s, &l, m.ident, best.harness);
    lock_hill();
    char *tree = best_tree(s, commit);
    build(s, tree, "best");
    build(s, root_dir, "candidate");
    Strs names = instrument_names(s);
    Judged j = {jnew(JOBJ), 0, 0};
    for (size_t ni = 0; ni < names.n && !j.losses; ni++) {
        Strs one = {0};
        spush(&one, names.v[ni]);
        Sample *samples = exact_only(s, names.v[ni]) ? once(s, tree, root_dir, names.v[ni], known_best(s, &l, m.ident, best.harness, best.product, names.v[ni]))
                                                   : paired(s, tree, root_dir, "best", "candidate", "blocks", BLOCKS, &one);
        Judged part = judge(s, samples, noise);
        for (size_t r = 0; r < part.rows->len; r++) jset(j.rows, part.rows->k[r], part.rows->v[r]);
        j.wins += part.wins;
        j.losses += part.losses;
    }
    unlock_hill();
    Pairs now = files_of(s);
    for (size_t i = 0; i < cand_files.n; i++) {
        Pair *found = bsearch(&cand_files.v[i], now.v, now.n, sizeof *now.v, pair_cmp);
        if (!found) stop("ERROR", "the working tree changed while it was being measured; gate it again once it is still");
    }
    const char *verdict = j.wins && !j.losses ? "PROMOTE" : j.losses ? "REJECT" : "TIE";
    J *values = jnew(JOBJ);
    for (size_t r = 0; r < j.rows->len; r++) jset(values, j.rows->k[r], jget(j.rows->v[r], "candidate"));
    J *entry = entry_of("verdict", &m);
    jset(entry, "best_commit", jstr(commit));
    jset(entry, "best", jstr(best.product));
    jset(entry, "candidate", jstr(cand.product));
    jset(entry, "harness", jstr(best.harness));
    jset(entry, "verdict", jstr(verdict));
    jset(entry, "rows", j.rows);
    jset(entry, "values", values);
    append(entry);
    puts(report(s, j.rows, verdict));
    return strcmp(verdict, "PROMOTE") ? 1 : 0;
}

static void loosened(Spec *old, Spec *new, Strs *found, Strs *held) {
    J *orew = jget(old->spec, "rewards"), *nrew = jget(new->spec, "rewards");
    for (size_t i = 0; i < orew->len; i++) {
        if (!listed(held, orew->k[i])) continue;
        J *n = jget(nrew, orew->k[i]);
        if (!n) {
            spush(found, fmtstr("%s was dropped", orew->k[i]));
            continue;
        }
        if (strcmp(jtext_or(n, "better", ""), jtext_or(orew->v[i], "better", ""))) spush(found, fmtstr("%s changed direction", orew->k[i]));
        if (jtruthy(jget(orew->v[i], "exact")) && !jtruthy(jget(n, "exact")))
            spush(found, fmtstr("%s went from exact to judged within a floor", orew->k[i]));
    }
    J *oi = jget(old->spec, "instruments"), *ni = jget(new->spec, "instruments");
    for (size_t i = 0; i < oi->len; i++) {
        J *n = jget(ni, oi->k[i]);
        int kept = 0;
        for (size_t r = 0; r < orew->len; r++) kept |= is(orew->v[r], "instrument", oi->k[i]) && listed(held, orew->k[r]);
        if (kept && n && jint_or(n, "blocks", BLOCKS) < jint_or(oi->v[i], "blocks", BLOCKS)) spush(found, fmtstr("%s runs fewer blocks", oi->k[i]));
    }
}

static int cmd_check(void) {
    const char *names[] = {"HEAD", ":hill.json", "HEAD:hill.json", "HEAD:.hill/ledger.jsonl", ":.hill/ledger.jsonl"};
    Blobs blobs = cat_files(names, 5);
    char *commit = blobs.ok[0];
    if (!commit || !blobs.ok[1]) return 0;
    Spec s = load_spec(blobs.ok[1], blobs.len[1]);
    Strs problems = {0};
    if (blobs.ok[2] && blobs.len[2]) {
        Spec old = load_spec(blobs.ok[2], blobs.len[2]);
        Ledger hl = read_ledger_text(blobs.ok[3] ? blobs.ok[3] : "", blobs.ok[3] ? blobs.len[3] : 0);
        Strs held = {0};
        for (size_t i = 0; i < hl.n; i++) {
            J *v = jget(hl.v[i], "values");
            for (size_t k = 0; v && k < v->len; k++)
                if (!listed(&held, v->k[k])) spush(&held, v->k[k]);
        }
        loosened(&old, &s, &problems, &held);
    }
    const char *head_ledger = blobs.ok[3] ? blobs.ok[3] : "";
    size_t head_len = blobs.ok[3] ? blobs.len[3] : 0;
    const char *staged_ledger = blobs.ok[4] ? blobs.ok[4] : "";
    size_t staged_len = blobs.ok[4] ? blobs.len[4] : 0;
    if (staged_len < head_len || memcmp(staged_ledger, head_ledger, head_len)) spush(&problems, "the ledger is append-only, and this commit rewrites it");
    Digests best = digests(&s, commit), cand = digests(&s, "index");
    if (strcmp(cand.product, best.product)) {
        if (strcmp(cand.harness, best.harness))
            spush(&problems, "the harness and the measured code both change: land the harness change on its own");
        else {
            Ledger l = read_ledger_text(staged_ledger, staged_len);
            Strs idents = {0}, labels = {0};
            for (size_t i = 0; i < l.n; i++) {
                J *e = l.v[i];
                if (!is(e, "harness", best.harness)) continue;
                if ((is(e, "type", "record") && is(e, "product", best.product)) ||
                    (is(e, "type", "verdict") && (is(e, "best", best.product) || (is(e, "verdict", "PROMOTE") && is(e, "candidate", best.product))))) {
                    size_t k = 0;
                    while (k < idents.n && strcmp(idents.v[k], jtext(jget(e, "machine")))) k++;
                    if (k == idents.n) spush(&idents, (char *)jtext(jget(e, "machine"))), spush(&labels, (char *)jtext(jget(e, "label")));
                    else labels.v[k] = (char *)jtext(jget(e, "label"));
                }
            }
            J *machines = jget(s.spec, "machines");
            for (size_t m = 0; machines && machines->t == JOBJ && m < machines->len; m++)
                if (!listed(&idents, machines->k[m])) spush(&idents, machines->k[m]), spush(&labels, (char *)jtext(machines->v[m]));
            if (!idents.n) spush(&problems, "nothing has measured HEAD: run hill gate on every machine");
            size_t judged = 0, promoted = 0, rejected = 0;
            for (size_t k = 0; k < idents.n; k++) {
                J *found = NULL;
                for (size_t i = 0; i < l.n; i++) {
                    J *e = l.v[i];
                    if (is(e, "type", "verdict") && is(e, "machine", idents.v[k]) && is(e, "best", best.product) && is(e, "candidate", cand.product) &&
                        is(e, "harness", best.harness))
                        found = e;
                }
                if (!found) {
                    spush(&problems, fmtstr("no verdict for this change on %s: run hill gate there and stage .hill/ledger.jsonl", labels.v[k]));
                    continue;
                }
                judged++;
                promoted += is(found, "verdict", "PROMOTE");
                if (is(found, "verdict", "REJECT")) {
                    rejected++;
                    spush(&problems, fmtstr("%s judged this change REJECT", labels.v[k]));
                }
            }
            if (idents.n && judged == idents.n && !promoted && !rejected) spush(&problems, "no machine judged this change better");
        }
    } else if (strcmp(cand.harness, best.harness)) {
        fprintf(stderr, "hill: a harness change; after committing it, run hill record on every machine with a record\n");
    }
    if (problems.n) {
        fprintf(stderr, "hill: this commit can't land:\n");
        for (size_t i = 0; i < problems.n; i++) fprintf(stderr, "  - %s\n", problems.v[i]);
        return 1;
    }
    return 0;
}

typedef struct {
    char *label;
    J *values;
} Column;

static size_t values_at(Spec *s, const char *commit, Column **out) {
    Digests d = digests(s, commit);
    Ledger l = read_ledger();
    Strs idents = {0};
    Column *cols = calloc(l.n + 1, sizeof *cols);
    J *machines = jget(s->spec, "machines");
    for (size_t i = 0; i < l.n; i++) {
        J *e = l.v[i];
        if (!is(e, "harness", d.harness)) continue;
        if (!((is(e, "type", "record") && is(e, "product", d.product)) ||
              (is(e, "type", "verdict") && is(e, "verdict", "PROMOTE") && is(e, "candidate", d.product))))
            continue;
        const char *ident = jtext(jget(e, "machine"));
        size_t k = 0;
        while (k < idents.n && strcmp(idents.v[k], ident)) k++;
        if (k == idents.n) spush(&idents, (char *)ident);
        const char *named = jtext(jget(machines, ident));
        cols[k] = (Column){(char *)(named ? named : jtext(jget(e, "label"))), jget(e, "values")};
    }
    *out = cols;
    return idents.n;
}

static void ordered(Spec *s, const char *key, Strs *rids, Strs *out) {
    J *rewards = jget(s->spec, "rewards");
    char *plural = fmtstr("%ss", key);
    J *listed = jget(s->spec, plural);
    for (size_t i = 0; listed && listed->t == JARR && i < listed->len; i++) {
        const char *v = listed->v[i]->s;
        int any = 0;
        for (size_t r = 0; r < rids->n && !any; r++) {
            const char *x = jtext(jget(jget(rewards, rids->v[r]), key));
            any = x && !strcmp(x, v);
        }
        if (any) spush(out, (char *)v);
    }
    for (size_t r = 0; r < rids->n; r++) {
        const char *x = jtext(jget(jget(rewards, rids->v[r]), key));
        int seen = 0;
        for (size_t k = 0; k < out->n; k++) seen |= (x == NULL && out->v[k] == NULL) || (x && out->v[k] && !strcmp(out->v[k], x));
        if (!seen) spush(out, (char *)x);
    }
}

static int same(const char *a, const char *b) { return (!a && !b) || (a && b && !strcmp(a, b)); }

static void table_lines(Buf *b, Strs *header, Strs *rows, size_t ncols) {
    bputs(b, "|");
    for (size_t i = 0; i < header->n; i++) bprintf(b, " %s |", header->v[i]);
    bputs(b, "\n| --- | :---: |");
    for (size_t i = 2; i < header->n; i++) bputs(b, " ---: |");
    bputc(b, '\n');
    for (size_t r = 0; r < rows->n / ncols; r++) {
        bputs(b, "|");
        for (size_t c = 0; c < ncols; c++) bprintf(b, " %s |", rows->v[r * ncols + c]);
        bputc(b, '\n');
    }
}

static const char *arrow(Spec *s, const char *rid) {
    return is(jget(jget(s->spec, "rewards"), rid), "better", "higher") ? "\xe2\x86\x91" : "\xe2\x86\x93";
}

static char *render(Spec *s, Column *cols, size_t ncols) {
    J *rewards = jget(s->spec, "rewards");
    Strs rids = {0}, unmeasured = {0};
    for (size_t r = 0; r < rewards->len; r++) {
        int any = 0;
        for (size_t c = 0; c < ncols && !any; c++) any = jget(cols[c].values, rewards->k[r]) != NULL;
        spush(any ? &rids : &unmeasured, rewards->k[r]);
    }
    Buf b = {0};
    const char *heading = jtext_or(s->spec, "readme_heading", HEADING);
    bprintf(&b, "## %s\n\n%s\n\n", heading,
            jtext_or(s->spec, "readme_intro",
                     "Every number here is held by the gate: a change is promoted only when at least one of them gets "
                     "better and none gets worse, on every machine with a record. \xe2\x86\x91 higher is better, \xe2\x86\x93 lower is better."));
    Strs groups = {0};
    ordered(s, "group", &rids, &groups);
    for (size_t gi = 0; gi < groups.n; gi++) {
        const char *group = groups.v[gi];
        Strs cased = {0}, plain_r = {0};
        for (size_t r = 0; r < rids.n; r++) {
            J *rw = jget(rewards, rids.v[r]);
            if (!same(jtext(jget(rw, "group")), group)) continue;
            spush(jtruthy(jget(rw, "case")) ? &cased : &plain_r, rids.v[r]);
        }
        bprintf(&b, "### %s\n\n", group && group[0] ? group : "Rewards");
        Strs configs = {0};
        ordered(s, "config", &cased, &configs);
        for (size_t ci = 0; ci < configs.n; ci++) {
            const char *config = configs.v[ci];
            for (size_t c = 0; c < ncols; c++) {
                Strs these = {0};
                for (size_t r = 0; r < cased.n; r++)
                    if (same(jtext(jget(jget(rewards, cased.v[r]), "config")), config) && jget(cols[c].values, cased.v[r])) spush(&these, cased.v[r]);
                if (!these.n) continue;
                Strs cases = {0};
                ordered(s, "case", &these, &cases);
                if (config && config[0]) bprintf(&b, "**%s on %s**\n\n", config, cols[c].label);
                else bprintf(&b, "**%s**\n\n", cols[c].label);
                Strs header = {0}, rows = {0}, names = {0};
                spush(&header, ""), spush(&header, "");
                for (size_t k = 0; k < cases.n; k++) spush(&header, cases.v[k]);
                for (size_t r = 0; r < these.n; r++) {
                    const char *nm = jtext_or(jget(rewards, these.v[r]), "label", these.v[r]);
                    int seen = 0;
                    for (size_t k = 0; k < names.n; k++) seen |= !strcmp(names.v[k], nm);
                    if (!seen) spush(&names, (char *)nm);
                }
                for (size_t ni = 0; ni < names.n; ni++) {
                    const char *pick[64] = {0}, *first = NULL;
                    for (size_t k = 0; k < cases.n && k < 64; k++)
                        for (size_t r = 0; r < these.n; r++) {
                            J *rw = jget(rewards, these.v[r]);
                            if (!strcmp(jtext_or(rw, "label", these.v[r]), names.v[ni]) && same(jtext(jget(rw, "case")), cases.v[k])) pick[k] = these.v[r];
                        }
                    for (size_t r = 0; r < these.n && !first; r++)
                        if (!strcmp(jtext_or(jget(rewards, these.v[r]), "label", these.v[r]), names.v[ni])) first = these.v[r];
                    spush(&rows, names.v[ni]);
                    spush(&rows, (char *)arrow(s, first));
                    for (size_t k = 0; k < cases.n; k++) spush(&rows, pick[k] ? fmt_value(s, pick[k], jget(cols[c].values, pick[k]), 0) : "—");
                }
                table_lines(&b, &header, &rows, header.n);
                bputc(&b, '\n');
            }
        }
        if (plain_r.n) {
            Strs pconfigs = {0};
            ordered(s, "config", &plain_r, &pconfigs);
            const char *combo_config[256];
            size_t combo_col[256], ncombo = 0;
            for (size_t ci = 0; ci < pconfigs.n; ci++)
                for (size_t c = 0; c < ncols; c++) {
                    int any = 0;
                    for (size_t r = 0; r < plain_r.n && !any; r++)
                        any = same(jtext(jget(jget(rewards, plain_r.v[r]), "config")), pconfigs.v[ci]) && jget(cols[c].values, plain_r.v[r]);
                    if (any && ncombo < 256) combo_config[ncombo] = pconfigs.v[ci], combo_col[ncombo++] = c;
                }
            Strs header = {0}, rows = {0}, names = {0};
            spush(&header, ""), spush(&header, "");
            for (size_t k = 0; k < ncombo; k++)
                spush(&header, combo_config[k] && combo_config[k][0] ? fmtstr("%s \xc2\xb7 %s", combo_config[k], cols[combo_col[k]].label)
                                                                     : cols[combo_col[k]].label);
            for (size_t r = 0; r < plain_r.n; r++) {
                const char *nm = jtext_or(jget(rewards, plain_r.v[r]), "label", plain_r.v[r]);
                int seen = 0;
                for (size_t k = 0; k < names.n; k++) seen |= !strcmp(names.v[k], nm);
                if (!seen) spush(&names, (char *)nm);
            }
            for (size_t ni = 0; ni < names.n; ni++) {
                const char *first = NULL;
                for (size_t r = 0; r < plain_r.n && !first; r++)
                    if (!strcmp(jtext_or(jget(rewards, plain_r.v[r]), "label", plain_r.v[r]), names.v[ni])) first = plain_r.v[r];
                spush(&rows, names.v[ni]);
                spush(&rows, (char *)arrow(s, first));
                for (size_t k = 0; k < ncombo; k++) {
                    const char *pick = NULL;
                    for (size_t r = 0; r < plain_r.n; r++) {
                        J *rw = jget(rewards, plain_r.v[r]);
                        if (!strcmp(jtext_or(rw, "label", plain_r.v[r]), names.v[ni]) && same(jtext(jget(rw, "config")), combo_config[k])) pick = plain_r.v[r];
                    }
                    J *v = pick ? jget(cols[combo_col[k]].values, pick) : NULL;
                    spush(&rows, v ? fmt_value(s, pick, v, 0) : "—");
                }
            }
            table_lines(&b, &header, &rows, header.n);
            bputc(&b, '\n');
        }
    }
    if (unmeasured.n) {
        bputs(&b, "### Not measured yet\n\n");
        Strs header = {0}, rows = {0};
        spush(&header, ""), spush(&header, "");
        for (size_t r = 0; r < unmeasured.n; r++) spush(&rows, name_of(s, unmeasured.v[r])), spush(&rows, (char *)arrow(s, unmeasured.v[r]));
        table_lines(&b, &header, &rows, 2);
        bputc(&b, '\n');
    }
    if (b.n && b.p[b.n - 1] == '\n') b.p[--b.n] = 0;
    return bstr(&b);
}

static int cmd_table(Spec *s) {
    char *commit = head_commit();
    if (!commit) stop("REFUSE", "commit something first: the table shows HEAD's record");
    Column *cols;
    size_t n = values_at(s, commit, &cols);
    char *section = render(s, cols, n);
    const char *readme = jtext_or(s->spec, "readme", "README.md");
    char *path = fmtstr("%s/%s", root_dir, readme);
    size_t len = 0;
    char *text = read_file(path, &len);
    if (!text) text = "", len = 0;
    char *heading = fmtstr("## %s", jtext_or(s->spec, "readme_heading", HEADING));
    Strs lines = {0};
    char *copy = strndup(text, len);
    for (char *p = copy;;) {
        char *nl = strchr(p, '\n');
        if (nl) *nl = 0;
        spush(&lines, p);
        if (!nl) break;
        p = nl + 1;
    }
    size_t start = lines.n;
    for (size_t i = 0; i < lines.n; i++) {
        char *t = strdup(lines.v[i]);
        if (!strcmp(strip(t), heading)) {
            start = i;
            break;
        }
    }
    Buf out = {0};
    if (start == lines.n) {
        size_t n2 = len;
        while (n2 && text[n2 - 1] == '\n') n2--;
        bput(&out, text, n2);
        char *t = strdup(text);
        if (strip(t)[0]) bputs(&out, "\n\n");
        bputs(&out, section);
        bputc(&out, '\n');
    } else {
        size_t end = lines.n;
        for (size_t i = start + 1; i < lines.n; i++)
            if (!strncmp(lines.v[i], "## ", 3)) {
                end = i;
                break;
            }
        for (size_t i = 0; i < start; i++) bputs(&out, lines.v[i]), bputc(&out, '\n');
        bputs(&out, section);
        for (size_t i = end; i < lines.n; i++) bputc(&out, '\n'), bputs(&out, lines.v[i]);
    }
    FILE *f = fopen(path, "w");
    if (!f) stop("ERROR", "cannot write %s", path);
    fwrite(out.p, 1, out.n, f);
    fclose(f);
    printf("hill: wrote the %s section of %s\n", heading + 3, strrchr(path, '/') + 1);
    return 0;
}

static int cmd_hook(void) {
    Proc r = spawn((char *[]){"git", "config", "core.hooksPath", NULL}, root_dir, NULL, 0, 1, 1, -1, NULL);
    char *configured = strip(r.out.p);
    char *hooks = configured[0] ? (configured[0] == '/' ? strdup(configured) : fmtstr("%s/%s", root_dir, configured))
                                : fmtstr("%s/hooks", strip(GIT("rev-parse", "--absolute-git-dir")));
    mkdir(hooks, 0777);
    char *hook = fmtstr("%s/pre-commit", hooks);
    const char *lines = "sh .hill/hill check || exit 1\n";
    char *existing = read_file(hook, NULL);
    if (existing) {
        if (!strstr(existing, ".hill/hill"))
            stop("REFUSE", "%s already exists; add these lines near its top:\n%s", hook, lines);
        return 0;
    }
    FILE *f = fopen(hook, "w");
    if (!f) stop("ERROR", "cannot write %s", hook);
    fprintf(f, "#!/bin/sh\n%s", lines);
    fclose(f);
    chmod(hook, 0755);
    hill_dir();
    printf("hill: installed %s\n", hook);
    return 0;
}

static char *find_root(void) {
    if (!getenv("GIT_DIR") && !getenv("GIT_WORK_TREE")) {
        char *dir = getcwd(NULL, 0);
        while (dir) {
            char *dotgit = fmtstr("%s/.git", dir);
            int found = !access(dotgit, F_OK);
            free(dotgit);
            if (found) return dir;
            char *slash = strrchr(dir, '/');
            if (!slash) break;
            if (slash == dir) {
                if (!dir[1]) break;
                dir[1] = 0;
            } else
                *slash = 0;
        }
    }
    Proc top = spawn((char *[]){"git", "rev-parse", "--show-toplevel", NULL}, NULL, NULL, 0, 1, 1, -1, NULL);
    if (top.status) stop("ERROR", "not inside a git repository");
    return strdup(strip(top.out.p));
}

int main(int argc, char **argv) {
    const char *commands[] = {"calibrate", "record", "gate", "table", "check", "hook"};
    int known = 0;
    for (int i = 0; argc == 2 && i < 6; i++) known |= !strcmp(argv[1], commands[i]);
    if (!known) {
        fprintf(stderr, "usage: hill calibrate | record | gate | table | check | hook\n");
        return 2;
    }
    if (setjmp(stop_env)) {
        fflush(stdout);
        fprintf(stderr, "%s: %s\n", stop_kind, stop_text);
        return 2;
    }
    root_dir = find_root();
    if (!strcmp(argv[1], "check")) return cmd_check();
    if (!strcmp(argv[1], "hook")) return cmd_hook();
    Spec s = load_spec_file();
    if (!strcmp(argv[1], "calibrate")) return cmd_calibrate(&s);
    if (!strcmp(argv[1], "record")) return cmd_record(&s);
    if (!strcmp(argv[1], "gate")) return cmd_gate(&s);
    return cmd_table(&s);
}

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
#include <signal.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define ALPHA 0.05
#define BLOCKS 2
#define BUDGET_S 5
#define RING 8
#define REFRESH_S 10
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
    Strs measured, harness, apart;
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
    J *ap = jget(s.spec, "not_measured");
    for (size_t i = 0; ap && ap->t == JARR && i < ap->len; i++) spush(&s.apart, ap->v[i]->s);
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
        if (under(p, &s->measured) && !under(p, &s->harness) && !under(p, &s->apart)) ppush(&product, p, everything->v[i].sha);
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
            if (!under(m.v[i].path, &s->harness) && !under(m.v[i].path, &s->apart)) ppush(&product, m.v[i].path, m.v[i].sha);
        harness = listing(tree, &s->harness);
    }
    d.product = pairs_digest(&product);
    d.harness = pairs_digest(&harness);
    return d;
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
    char *ignore = fmtstr("%s/.gitignore", d);
    default_file(ignore, "trees/\nlogs/\nrun/\nbin.path\n");
    char *have = read_file(ignore, NULL);
    FILE *f = have && !strstr(have, "run/") ? fopen(ignore, "a") : NULL;
    if (f) fputs(have[0] && have[strlen(have) - 1] != '\n' ? "\nrun/\n" : "run/\n", f), fclose(f);
    mkdir(fmtstr("%s/run", d), 0777);
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

static char *tree_at(Spec *s, const char *commit, const char *slot) {
    char *trees = fmtstr("%s/trees", hill_dir());
    char *path = fmtstr("%s/%s", trees, slot);
    if (access(fmtstr("%s/.git", path), F_OK)) {
        mkdir(trees, 0777);
        Strs old = entries_of(trees);
        for (size_t i = 0; i < old.n; i++) {
            if (!strcmp(old.v[i], "a") || !strcmp(old.v[i], "b")) continue;
            char *gone = fmtstr("%s/%s", trees, old.v[i]);
            char *argv[] = {"git", "worktree", "remove", "--force", gone, NULL};
            spawn(argv, root_dir, NULL, 0, 1, 1, -1, NULL);
            char *rm[] = {"rm", "-rf", gone, NULL};
            if (!access(gone, F_OK)) spawn(rm, root_dir, NULL, 0, 1, 1, -1, NULL);
        }
        char *rm[] = {"rm", "-rf", path, NULL};
        spawn(rm, root_dir, NULL, 0, 1, 1, -1, NULL);
        GIT("worktree", "prune");
        GIT("worktree", "add", "--detach", "--force", path, commit);
    } else
        GIT("-C", path, "checkout", "-q", "--detach", "-f", commit);
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

static int listed(Strs *names, const char *name) {
    for (size_t i = 0; i < names->n; i++)
        if (!strcmp(names->v[i], name)) return 1;
    return 0;
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

typedef struct {
    J *rows;
    int wins, losses;
} Judged;

static char *report(Spec *s, J *rows, const char *verdict, const char *label) {
    Buf b = {0}, named = {0}, table = {0};
    const char *pick = !strcmp(verdict, "PROMOTE") ? "win" : !strcmp(verdict, "REJECT") ? "loss" : "";
    int counts[3] = {0};
    for (size_t i = 0; i < rows->len; i++) {
        J *row = rows->v[i];
        J *change = jget(row, "change");
        const char *result = jget(row, "result")->s;
        counts[!strcmp(result, "win") ? 0 : !strcmp(result, "tie") ? 1 : 2]++;
        if (!strcmp(result, "tie") && num_eq(jget(row, "best"), jget(row, "candidate"))) continue;
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
        bprintf(&table, "\n| %s | %s | %s | %s | %s |", name_of(s, rows->k[i]), fa, fc, bstr(&c), result);
        if (!strcmp(result, pick)) bprintf(&named, "\n  %s: %s \xe2\x86\x92 %s%s%s%s", name_of(s, rows->k[i]), fa, fc, d.n ? " (" : "", bstr(&d), d.n ? ")" : "");
    }
    bprintf(&b, "%s:", label);
    if (table.n) bprintf(&b, "\n| Reward | Best | Candidate | Change | Verdict |\n| --- | ---: | ---: | ---: | --- |%s", bstr(&table));
    size_t total = jget(s->spec, "rewards")->len;
    if (rows->len < total) bprintf(&b, "\nThe step stopped at the first loss: %zu of %zu rewards were not measured.", total - rows->len, total);
    if (!strcmp(verdict, "PROMOTE"))
        bprintf(&b, "\nPROMOTE on %d reward%s, with %d equal and none worse:%s", counts[0], counts[0] == 1 ? "" : "s", counts[1], bstr(&named));
    else if (!strcmp(verdict, "REJECT"))
        bprintf(&b, "\nREJECT on %d reward%s, with %d better and %d equal:%s", counts[2], counts[2] == 1 ? "" : "s", counts[0], counts[1], bstr(&named));
    else
        bprintf(&b, "\nTIE: %d better, %d equal, %d worse.", counts[0], counts[1], counts[2]);
    return bstr(&b);
}

typedef struct {
    char *ident, *label;
} Machine;

typedef struct {
    long long vram;
    char *id, *name;
} Card;

static int card_cmp(const void *a, const void *b) {
    const Card *x = a, *y = b;
    if (x->vram != y->vram) return x->vram < y->vram ? 1 : -1;
    return strcmp(x->id, y->id);
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
    mem->i = (kb + (1LL << 19)) >> 20;
    J *cards = jnew(JARR);
    Strs names = entries_of("/sys/class/drm");
    Card *found = malloc((names.n + 1) * sizeof *found);
    size_t nfound = 0;
    char *lspci = on_path("lspci");
    for (size_t i = 0; i < names.n; i++) {
        struct stat st;
        const char *n = names.v[i];
        char *device = fmtstr("/sys/class/drm/%s/device", n);
        if (strncmp(n, "card", 4) || !n[4] || n[4 + strspn(n + 4, "0123456789")] || lstat(device, &st)) continue;
        char *text = read_file(fmtstr("%s/mem_info_vram_total", device), NULL), *end = NULL;
        long long vram = text ? strtoll(strip(text), &end, 10) : 0;
        if (end && *end) vram = 0;
        char *vendor = read_file(fmtstr("%s/vendor", device), NULL), *product = read_file(fmtstr("%s/device", device), NULL);
        char *real = realpath(device, NULL);
        char *name = strdup(real ? strrchr(real, '/') + 1 : device);
        if (lspci) {
            char *argv[] = {lspci, "-mm", "-s", name, NULL};
            Proc r = spawn(argv, NULL, NULL, 0, 1, 1, -1, NULL);
            Strs fields = shlex(r.out.n ? r.out.p : "\"\"");
            if (fields.n > 3) name = fields.v[3];
        }
        found[nfound++] = (Card){vram, fmtstr("%s:%s", vendor ? strip(vendor) : "?", product ? strip(product) : "?"), name};
    }
    qsort(found, nfound, sizeof *found, card_cmp);
    for (size_t i = 0; i < nfound; i++) {
        J *card = jnew(JOBJ), *gb = jnew(JINT);
        gb->i = found[i].vram >> 30;
        jset(card, "id", jstr(found[i].id));
        jset(card, "vram_gb", gb);
        jpush(cards, card);
    }
    jset(info, "cpu", jstr(cpu));
    jset(info, "memory_gb", mem);
    jset(info, "gpus", cards);
    label = nfound && found[0].vram >> 30 >= 2 ? found[0].name : cpu;
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

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

static int read_line(int fd, Buf *got, double deadline, char **line) {
    for (;;) {
        char *nl = got->n ? memchr(got->p, '\n', got->n) : NULL;
        if (nl) {
            size_t n = (size_t)(nl - got->p);
            *line = strndup(got->p, n);
            memmove(got->p, nl + 1, got->n - n - 1);
            got->n -= n + 1;
            return 1;
        }
        int ms = -1;
        if (deadline > 0) {
            double left = deadline - now_s();
            if (left <= 0) return -1;
            ms = (int)(left * 1000) + 1;
        }
        struct pollfd p = {fd, POLLIN, 0};
        int r = poll(&p, 1, ms);
        if (r < 0 && errno == EINTR) continue;
        if (r == 0) return -1;
        char chunk[65536];
        ssize_t k = read(fd, chunk, sizeof chunk);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) return 0;
        bput(got, chunk, (size_t)k);
    }
}

static pid_t launch(const char *cmd, const char *cwd, const char *side, int *in, int *out, const char *log) {
    int pin[2], pout[2];
    if (pipe2(pin, O_CLOEXEC) || pipe2(pout, O_CLOEXEC)) stop("ERROR", "pipe: %s", strerror(errno));
    int fd = open(log, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0666);
    char **env = env_with(side, cwd);
    fflush(stdout);
    fflush(stderr);
    pid_t pid = fork();
    if (pid < 0) stop("ERROR", "fork: %s", strerror(errno));
    if (!pid) {
        setpgid(0, 0);
        if (chdir(cwd)) _exit(127);
        dup2(in ? pin[0] : open("/dev/null", O_RDONLY), 0);
        dup2(pout[1], 1);
        dup2(fd, 2);
        environ = env;
        execl("/bin/sh", "sh", "-c", cmd, (char *)NULL);
        _exit(127);
    }
    setpgid(pid, pid);
    close(pin[0]);
    close(pout[1]);
    close(fd);
    if (in) *in = pin[1];
    else close(pin[1]);
    *out = pout[0];
    return pid;
}

static int reap(pid_t pid) {
    int st = 0;
    while (waitpid(pid, &st, 0) < 0 && errno == EINTR) {
    }
    return WIFEXITED(st) ? WEXITSTATUS(st) : -WTERMSIG(st);
}

typedef struct {
    char *serve, *tree, *excl, *log;
    pid_t pid;
    int in, out;
    Buf got;
} Worker;

static Worker *workers;
static size_t nworkers;

static void worker_kill(Worker *w) {
    if (!w->pid) return;
    close(w->in);
    kill(-w->pid, SIGTERM);
    int st, gone = 0;
    for (double until = now_s() + 30; !gone && now_s() < until;) {
        gone = waitpid(w->pid, &st, WNOHANG) != 0;
        if (!gone) poll(NULL, 0, 20);
    }
    kill(-w->pid, SIGKILL);
    close(w->out);
    if (!gone) reap(w->pid);
    w->pid = 0;
    w->got.n = 0;
}

static Worker *worker_for(const char *serve, const char *tree, const char *side, const char *excl, const char *name) {
    for (size_t i = 0; i < nworkers; i++)
        if (workers[i].pid && !strcmp(workers[i].serve, serve) && !strcmp(workers[i].tree, tree)) return &workers[i];
    for (size_t i = 0; excl && i < nworkers; i++)
        if (workers[i].pid && workers[i].excl && !strcmp(workers[i].excl, excl)) worker_kill(&workers[i]);
    Worker *w = NULL;
    for (size_t i = 0; i < nworkers && !w; i++)
        if (!workers[i].pid) w = &workers[i];
    if (!w) workers = realloc(workers, (nworkers + 1) * sizeof *workers), w = &workers[nworkers++];
    *w = (Worker){strdup(serve), strdup(tree), excl ? strdup(excl) : NULL, fmtstr("%s/%s-%s.log", logs_dir(), name, side), 0, -1, -1, {0}};
    printf("hill: %s: starting the %s's worker\n", name, side);
    w->pid = launch(serve, tree, side, &w->in, &w->out, w->log);
    for (;;) {
        char *line;
        if (read_line(w->out, &w->got, 0, &line) <= 0) {
            worker_kill(w);
            stop("ERROR", "%s's worker for the %s exited before it was ready (see %s)", name, side, w->log);
        }
        J *j = parse(line, strlen(line));
        if (j && j->t == JOBJ && jtruthy(jget(j, "ready"))) return w;
    }
}

static double budget_of(J *inst) {
    J *b = jget(inst, "budget_s");
    return jnum(b) && jd(b) > 0 ? jd(b) : BUDGET_S;
}

static void over(const char *name, double budget) { stop("ERROR", "%s took more than %g s: make it fast first", name, budget); }

static char *one_shot(const char *cmd, const char *tree, const char *side, const char *name, double budget, const char *log, int *status) {
    int out;
    pid_t pid = launch(cmd, tree, side, NULL, &out, log);
    Buf all = {0};
    char chunk[65536];
    double deadline = now_s() + budget;
    int done = 0;
    while (!done) {
        double left = deadline - now_s();
        struct pollfd p = {out, POLLIN, 0};
        int q = left > 0 ? poll(&p, 1, (int)(left * 1000) + 1) : 0;
        if (q < 0 && errno == EINTR) continue;
        if (q == 0) {
            kill(-pid, SIGKILL);
            close(out);
            reap(pid);
            over(name, budget);
        }
        ssize_t k = read(out, chunk, sizeof chunk);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) done = 1;
        else bput(&all, chunk, (size_t)k);
    }
    close(out);
    *status = reap(pid);
    return bstr(&all);
}

static char *last_line(char *text, size_t n) {
    char *last = NULL;
    size_t lastlen = 0, i = 0;
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
    return last ? strndup(last, lastlen) : NULL;
}

static J *last_detail, *cand_detail;

static int caller_fd = -1;

static void caller_here(void) {
    if (caller_fd < 0) return;
    struct pollfd p = {caller_fd, POLLIN, 0};
    char c;
    if (poll(&p, 1, 0) > 0 && recv(caller_fd, &c, 1, MSG_PEEK | MSG_DONTWAIT) == 0) stop("ERROR", "the step's caller left, so the step ends");
}

static J *run_inst(Spec *s, const char *name, const char *tree, const char *side) {
    caller_here();
    J *inst = jget(jget(s->spec, "instruments"), name);
    J *need = jget(inst, "memory_gb");
    if (jnum(need) && jd(need) != 0 && available_gb() < jd(need)) {
        Buf b = {0};
        dump(&b, need);
        stop("REFUSE", "%s needs %s GB of free memory and %.1f GB is free; nothing was started", name, bstr(&b), available_gb());
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
    double budget = budget_of(inst);
    const char *serve = jtext(jget(inst, "serve"));
    char *line = NULL, *log;
    if (serve) {
        Worker *w = worker_for(serve, tree, side, jtext(jget(inst, "exclusive")), name);
        log = w->log;
        char *req = fmtstr("run %s\n", name);
        ssize_t sent = write(w->in, req, strlen(req));
        (void)sent;
        int got = read_line(w->out, &w->got, now_s() + budget, &line);
        if (got <= 0) worker_kill(w);
        if (got < 0) over(name, budget);
        if (!got) stop("ERROR", "%s's worker for the %s exited (see %s)", name, side, log);
    } else {
        log = fmtstr("%s/%s-%s.log", logs_dir(), name, side);
        int status;
        char *text = one_shot(jtext(jget(inst, "run")), tree, side, name, budget, log, &status);
        if (status) stop("ERROR", "%s failed on the %s with exit %d (see %s)", name, side, status, log);
        line = last_line(text, strlen(text));
    }
    J *values = line ? parse(line, strlen(line)) : NULL;
    if (!values || values->t != JOBJ) stop("ERROR", "%s did not answer with one JSON object (see %s)", name, log);
    last_detail = jget(values, "detail");
    return values;
}

static char *fingerprint(Spec *s, const char *name, const char *tree, const char *side) {
    J *inst = jget(jget(s->spec, "instruments"), name);
    const char *cmd = jtext(jget(inst, "depends"));
    if (!cmd || !cmd[0]) return NULL;
    int status;
    char *text = one_shot(cmd, tree, side, fmtstr("%s's depends", name), budget_of(inst), fmtstr("%s/%s-depends-%s.log", logs_dir(), name, side), &status);
    text = strip(text);
    return !status && text[0] ? text : NULL;
}

typedef struct {
    J *v;
    double at;
} Kept;

typedef struct {
    Kept k[RING];
    size_t n;
} Ring;

static struct {
    Spec s;
    char *commit, *tree;
    Ring *rings;
    J *fps, *details;
    double *next;
} B;

static void ring_push(Ring *r, J *v) {
    if (r->n == RING) memmove(r->k, r->k + 1, (RING - 1) * sizeof *r->k), r->n--;
    r->k[r->n++] = (Kept){v, now_s()};
}

static int of(Spec *s, size_t r, const char *name) { return !strcmp(jget(jget(s->spec, "rewards")->v[r], "instrument")->s, name); }

static size_t held(Spec *s, const char *name) {
    J *rewards = jget(s->spec, "rewards");
    size_t least = RING;
    for (size_t r = 0; r < rewards->len; r++)
        if (of(s, r, name) && B.rings[r].n < least) least = B.rings[r].n;
    return least;
}

static J *best_sample(Spec *s, const char *name, Vals *into) {
    J *v = run_inst(s, name, B.tree, "best");
    J *rewards = jget(s->spec, "rewards");
    for (size_t r = 0; r < rewards->len; r++)
        if (of(s, r, name)) {
            J *x = field(s, rewards->k[r], v, name);
            ring_push(&B.rings[r], x);
            if (into) vpush(&into[r], x);
        }
    jset(B.details, name, last_detail ? last_detail : jnew(JNULL));
    return v;
}

static void take(Spec *s, const char *name, const char *tree, Vals *into) {
    J *v = run_inst(s, name, tree, "candidate");
    J *rewards = jget(s->spec, "rewards");
    for (size_t r = 0; r < rewards->len; r++)
        if (of(s, r, name)) vpush(&into[r], field(s, rewards->k[r], v, name));
    cand_detail = last_detail;
}

static Vals ring_vals(Ring *r) {
    Vals v = {0};
    for (size_t i = 0; i < r->n; i++) vpush(&v, r->k[i].v);
    return v;
}

static Noise ring_noise(const char *rid, Ring *r) {
    if (r->n < 2) stop("ERROR", "%s needs two samples of the best to know its noise", rid);
    double sum = 0;
    for (size_t i = 1; i < r->n; i++) {
        double a = jd(r->k[i - 1].v), b = jd(r->k[i].v);
        if (a <= 0 || b <= 0) stop("ERROR", "%s is noisy but not positive; it has to be exact", rid);
        sum += log(b / a) * log(b / a);
    }
    return (Noise){sqrt(sum / (double)(r->n - 1) / 2), (long long)(r->n - 1), 1, 0};
}

static void judge_rows(Spec *s, const char *name, Vals *a, Vals *b, J *rows, int *wins, int *losses) {
    J *rewards = jget(s->spec, "rewards");
    long long noisy = 0;
    for (size_t i = 0; i < rewards->len; i++) noisy += !jtruthy(jget(rewards->v[i], "exact"));
    for (size_t r = 0; r < rewards->len; r++) {
        if (!of(s, r, name)) continue;
        const char *rid = rewards->k[r];
        J *spec = rewards->v[r], *row = jnew(JOBJ);
        double sign = is(spec, "better", "higher") ? 1 : -1;
        const char *result;
        int exact = jtruthy(jget(spec, "exact")), gone = nulls(rid, &a[r]), missing = nulls(rid, &b[r]);
        if (gone || missing) {
            jset(row, "best", gone ? jnew(JNULL) : exact ? exact_value(rid, &a[r]) : median(&a[r]));
            jset(row, "candidate", missing ? jnew(JNULL) : exact ? exact_value(rid, &b[r]) : median(&b[r]));
            jset(row, "change", jnew(JNULL));
            result = gone && missing ? "tie" : gone ? "win" : "loss";
        } else if (exact) {
            J *x = exact_value(rid, &a[r]), *c = exact_value(rid, &b[r]);
            jset(row, "best", x);
            jset(row, "candidate", c);
            int zero = x->t == JINT ? x->i == 0 : x->d == 0;
            jset(row, "change", zero ? jnew(JNULL) : jreal(jd(c) / jd(x) - 1));
            double diff = (x->t == JINT && c->t == JINT) ? (double)(c->i - x->i) : jd(c) - jd(x);
            result = diff * sign > 0 ? "win" : diff * sign < 0 ? "loss" : "tie";
        } else {
            double ma = fmean_vals(&a[r]), mc = fmean_vals(&b[r]);
            if (ma <= 0 || mc <= 0) stop("ERROR", "%s is noisy but not positive; it has to be exact", rid);
            double d = log(mc / ma);
            Noise nz = ring_noise(rid, &B.rings[r]);
            double fl = quantile(1 - ALPHA / (double)(noisy > 1 ? noisy : 1), nz.dof) * nz.sigma * sqrt(1.0 / (double)a[r].n + 1.0 / (double)b[r].n);
            jset(row, "best", median(&a[r]));
            jset(row, "candidate", median(&b[r]));
            jset(row, "change", jreal(expm1(d)));
            jset(row, "floor", jreal(expm1(fl)));
            result = d * sign > fl ? "win" : d * sign < -fl ? "loss" : "tie";
        }
        jset(row, "result", jstr(result));
        jset(rows, rid, row);
        *wins += !strcmp(result, "win");
        *losses += !strcmp(result, "loss");
    }
}

static void end_best(void) {
    for (size_t i = 0; i < nworkers; i++) worker_kill(&workers[i]);
    B.commit = NULL;
}

static Spec spec_at(const char *commit) {
    const char *names[] = {fmtstr("%s:hill.json", commit)};
    Blobs b = cat_files(names, 1);
    if (!b.ok[0]) stop("ERROR", "%.12s has no hill.json", commit);
    return load_spec(b.ok[0], b.len[0]);
}

static void set_best(const char *commit) {
    if (B.commit && !strcmp(B.commit, commit)) return;
    Spec s = spec_at(commit);
    if (B.commit) {
        Digests was = digests(&B.s, B.commit), now = digests(&s, commit);
        if (!strcmp(was.product, now.product) && !strcmp(was.harness, now.harness)) {
            tree_at(&s, commit, B.tree + strlen(B.tree) - 1);
            B.commit = strdup(commit);
            return;
        }
    }
    end_best();
    char *tree = tree_at(&s, commit, B.tree && !strcmp(B.tree + strlen(B.tree) - 1, "b") ? "b" : "a");
    build(&s, tree, "best");
    size_t n = jget(s.spec, "rewards")->len;
    B.s = s;
    B.tree = tree;
    B.rings = calloc(n + 1, sizeof *B.rings);
    B.fps = jnew(JOBJ);
    B.details = jnew(JOBJ);
    B.next = calloc(jget(s.spec, "instruments")->len + 1, sizeof *B.next);
    B.commit = strdup(commit);
}

static char *git_path(const char *what) {
    char *p = strip(GIT("rev-parse", "--git-path", what));
    return p[0] == '/' ? p : fmtstr("%s/%s", root_dir, p);
}

static char *run_path(const char *name) { return fmtstr("%s/run/%s", hill_dir(), name); }

static void copy_file(const char *from, const char *to) {
    size_t n = 0;
    char *text = read_file(from, &n);
    FILE *f = fopen(to, "wb");
    if (!f) stop("ERROR", "cannot write %s", to);
    if (text) fwrite(text, 1, n, f);
    fclose(f);
}

static char *snapshot(void) {
    char *idx = run_path("index");
    struct stat st;
    copy_file(git_path("index"), idx);
    if (!stat(git_path("index"), &st)) utimensat(AT_FDCWD, idx, (struct timespec[]){
#ifdef __APPLE__
            st.st_atimespec, st.st_mtimespec
#else
            st.st_atim, st.st_mtim
#endif
        }, 0);
    setenv("GIT_INDEX_FILE", idx, 1);
    GIT("add", "-A");
    GIT("reset", "-q", "HEAD", "--", ".hill/ledger.jsonl");
    char *tree = strip(GIT("write-tree"));
    unsetenv("GIT_INDEX_FILE");
    return tree;
}

static char *commit_of(const char *tree, const char *parent, const char *message) {
    return strip(GIT("commit-tree", tree, "-p", parent, "-m", message));
}

static int cmd_check(void);

static char *land(const char *head, const char *tree, const char *message) {
    char *idx = run_path("index");
    setenv("GIT_INDEX_FILE", idx, 1);
    GIT("read-tree", tree);
    if (!access(fmtstr("%s/.hill/ledger.jsonl", root_dir), F_OK)) GIT("add", "-f", ".hill/ledger.jsonl");
    int bad = cmd_check();
    char *t = strip(GIT("write-tree"));
    unsetenv("GIT_INDEX_FILE");
    if (bad) stop("ERROR", "the commit would not pass the hook (above)");
    char *c = commit_of(t, head, message);
    GIT("update-ref", "-m", fmtstr("hill step: %s", message), "HEAD", c, head);
    GIT("reset", "-q");
    return c;
}

static char *slug(const char *message) {
    Buf b = {0};
    for (const char *p = message; *p && b.n < 60; p++) {
        if (isalnum((unsigned char)*p)) bputc(&b, (char)tolower((unsigned char)*p));
        else if (b.n && b.p[b.n - 1] != '-') bputc(&b, '-');
    }
    while (b.n && b.p[b.n - 1] == '-') b.n--;
    if (!b.n) bputs(&b, "candidate");
    char *base = bstr(&b), *ref = fmtstr("refs/candidates/%s", base);
    for (int i = 2;; i++) {
        char *argv[] = {"git", "rev-parse", "-q", "--verify", ref, NULL};
        if (spawn(argv, root_dir, NULL, 0, 1, 1, -1, NULL).status) return ref;
        ref = fmtstr("refs/candidates/%s-%d", base, i);
    }
}

static char *shelve(const char *head, const char *tree, const char *message, const char *why) {
    char *ref = slug(message);
    char *c = commit_of(tree, head, fmtstr("%s\n\n%s", message, why));
    GIT("update-ref", ref, c);
    if (!strcmp(snapshot(), tree)) {
        size_t n = 0;
        char *ledger = fmtstr("%s/.hill/ledger.jsonl", root_dir);
        char *kept = read_file(ledger, &n);
        GIT("reset", "-q", "--hard", "HEAD");
        GIT("clean", "-fdq");
        if (kept) {
            FILE *f = fopen(ledger, "wb");
            if (f) fwrite(kept, 1, n, f), fclose(f);
        }
        printf("hill: the working tree is back at HEAD\n");
    } else
        printf("hill: the working tree changed during the step, so it was left as it is\n");
    return ref;
}

typedef struct {
    char *name, *out;
    pid_t pid;
} Remote;

static Remote *remotes;
static size_t nremotes;

static char *quoted(const char *s) {
    Buf b = {0};
    bputc(&b, '\'');
    for (; *s; s++)
        if (*s == '\'') bputs(&b, "'\\''");
        else bputc(&b, *s);
    bputc(&b, '\'');
    return bstr(&b);
}

static void start_remotes(const char *sha) {
    const char *home = getenv("HOME"), *base = strrchr(root_dir, '/') ? strrchr(root_dir, '/') + 1 : root_dir;
    char *path = fmtstr("%s/.config/hill/%s.json", home ? home : "", base);
    size_t n;
    char *text = read_file(path, &n);
    if (!text) return;
    J *list = parse(text, n);
    if (!list || list->t != JARR) stop("ERROR", "%s is not a JSON list of machines", path);
    remotes = calloc(list->len + 1, sizeof *remotes);
    for (size_t i = 0; i < list->len; i++) {
        J *m = list->v[i], *ssh = jget(m, "ssh");
        const char *name = jtext(jget(m, "name")), *dir = jtext(jget(m, "dir")), *url = jtext(jget(m, "git"));
        if (!name || !dir || !url || !ssh || ssh->t != JARR) stop("ERROR", "%s: each machine needs name, ssh, dir and git", path);
        Buf cmd = {0};
        bprintf(&cmd, "git push -q %s %s && exec", quoted(url), quoted(fmtstr("+%s:refs/hill/candidate", sha)));
        for (size_t k = 0; k < ssh->len; k++) bprintf(&cmd, " %s", quoted(ssh->v[k]->s));
        bprintf(&cmd, " %s", quoted(jtext_or(m, "shell", "bash -s")));
        char *script = run_path(fmtstr("%s.sh", name));
        FILE *f = fopen(script, "w");
        if (!f) stop("ERROR", "cannot write %s", script);
        fprintf(f, "%s\ncd %s%s && .hill/hill step --remote %s\n", jtext_or(m, "pre", ""), strncmp(dir, "~/", 2) ? "" : "~/",
                quoted(strncmp(dir, "~/", 2) ? dir : dir + 2), sha);
        fclose(f);
        bprintf(&cmd, " < %s", quoted(script));
        Remote *r = &remotes[nremotes++];
        r->name = strdup(name);
        r->out = fmtstr("%s/%s.out", logs_dir(), name);
        int o;
        unlink(r->out);
        r->pid = launch(bstr(&cmd), root_dir, "remote", NULL, &o, r->out);
        close(o);
        printf("hill: %s: measuring there\n", name);
    }
}

static J *gather_remotes(int cut) {
    J *out = jnew(JARR);
    for (size_t i = 0; i < nremotes; i++) {
        Remote *r = &remotes[i];
        if (!r->pid) continue;
        if (cut) kill(-r->pid, SIGTERM);
        reap(r->pid);
        r->pid = 0;
        if (cut) continue;
        size_t n = 0;
        char *text = read_file(r->out, &n);
        J *e = NULL;
        for (char *line = text ? strtok(text, "\n") : NULL; line; line = strtok(NULL, "\n"))
            if (line[0] == '{') e = parse(line, strlen(line));
        if (!e || !is(e, "type", "verdict")) stop("ERROR", "%s gave no verdict (see %s)", r->name, r->out);
        jpush(out, e);
    }
    nremotes = 0;
    return out;
}

static void print_detail(Spec *s, J *rows, J *bdet, J *cdet) {
    const char *worst = NULL;
    double size = -1;
    for (size_t i = 0; i < rows->len; i++) {
        if (!is(rows->v[i], "result", "loss")) continue;
        J *c = jget(rows->v[i], "change");
        double x = c && c->t != JNULL ? fabs(jd(c)) : INFINITY;
        if (x > size) size = x, worst = rows->k[i];
    }
    if (!worst) return;
    const char *name = jget(jget(jget(s->spec, "rewards"), worst), "instrument")->s;
    J *bd = jget(bdet, name), *cd = jget(cdet, name);
    if ((!bd || bd->t != JOBJ) && (!cd || cd->t != JOBJ)) return;
    printf("\nWhere %s lost (%s):\n| | Best | Candidate |\n| --- | ---: | ---: |\n", name_of(s, worst), name);
    J *keys = bd && bd->t == JOBJ ? bd : cd;
    for (size_t k = 0; k < keys->len; k++) {
        Buf x = {0}, y = {0};
        J *a = jget(bd, keys->k[k]), *c = jget(cd, keys->k[k]);
        if (a) dump(&x, a);
        if (c) dump(&y, c);
        printf("| %s | %s | %s |\n", keys->k[k], a ? bstr(&x) : "", c ? bstr(&y) : "");
    }
}

static int step(const char *message, const char *remote_sha) {
    char *head, *cand, *tree = NULL;
    if (remote_sha) {
        cand = strip(GIT("rev-parse", "--verify", fmtstr("%s^{commit}", remote_sha)));
        head = strip(GIT("rev-parse", "--verify", fmtstr("%s^", cand)));
    } else {
        head = head_commit();
        if (!head) stop("REFUSE", "commit something first: a step is judged against HEAD");
        if (!message || !message[0]) stop("REFUSE", "a step needs a message");
    }
    set_best(head);
    Spec *s = &B.s;
    if (!remote_sha) {
        tree = snapshot();
        if (!strcmp(tree, strip(GIT("rev-parse", fmtstr("%s^{tree}", head))))) stop("REFUSE", "nothing changed since HEAD");
        cand = commit_of(tree, head, message);
    }
    Digests best = digests(s, head), now = digests(s, cand);
    int product = strcmp(now.product, best.product) != 0;
    if (product && strcmp(now.harness, best.harness)) stop("REFUSE", "the harness and the measured code both changed: step the harness change on its own first");
    if (!product) {
        if (remote_sha) stop("REFUSE", "nothing measured changed");
        char *c = land(head, tree, message);
        printf("hill: nothing measured changed, so nothing was measured: landed %.12s\n", c);
        return 0;
    }
    Machine m = machine(s);
    Ledger l = read_ledger();
    for (size_t i = 0; !remote_sha && i < l.n; i++) {
        J *e = l.v[i];
        if (is(e, "type", "verdict") && is(e, "machine", m.ident) && is(e, "best", best.product) && is(e, "candidate", now.product) &&
            is(e, "harness", best.harness))
            stop("REFUSE", "this candidate already has a verdict on this machine (%s); a new verdict needs a new candidate", jtext(jget(e, "verdict")));
    }
    if (!remote_sha) start_remotes(cand);
    J *rewards = jget(s->spec, "rewards");
    Strs names = instrument_names(s);
    J *rows = jnew(JOBJ), *cdet = jnew(JOBJ), *bdet = jnew(JOBJ), *fps = jnew(JOBJ), *carried = jnew(JOBJ);
    Vals *cv = calloc(rewards->len + 1, sizeof *cv);
    int wins = 0, losses = 0, built = 0;
    char *ctree = NULL;
    for (size_t ni = 0; ni < names.n && !losses; ni++) {
        const char *name = names.v[ni];
        int any = 0;
        for (size_t r = 0; r < rewards->len; r++) any |= of(s, r, name);
        if (!any) continue;
        J *inst = jget(jget(s->spec, "instruments"), name);
        if (!ctree) ctree = tree_at(s, cand, B.tree[strlen(B.tree) - 1] == 'a' ? "b" : "a");
        if (jtext(jget(inst, "depends"))) {
            if (!jget(B.fps, name)) {
                char *f = fingerprint(s, name, B.tree, "best");
                jset(B.fps, name, f ? jstr(f) : jnew(JNULL));
            }
            char *f = fingerprint(s, name, ctree, "candidate");
            if (f) jset(fps, name, jstr(f));
            if (f && is(B.fps, name, f)) {
                if (!held(s, name)) best_sample(s, name, NULL);
                printf("hill: %s: depends on the same as the best's (%.16s): its rewards carry over\n", name, f);
                for (size_t r = 0; r < rewards->len; r++) {
                    if (!of(s, r, name)) continue;
                    J *row = jnew(JOBJ), *v = B.rings[r].k[B.rings[r].n - 1].v;
                    jset(row, "best", v);
                    jset(row, "candidate", v);
                    jset(row, "change", jnew(JNULL));
                    jset(row, "result", jstr("tie"));
                    jset(rows, rewards->k[r], row);
                    vpush(&cv[r], v);
                }
                jset(carried, name, jstr(f));
                continue;
            }
        }
        if (!built) build(s, ctree, "candidate"), built = 1;
        Vals *a = calloc(rewards->len + 1, sizeof *a), *b = calloc(rewards->len + 1, sizeof *b);
        long long blocks = jint_or(inst, "blocks", BLOCKS);
        if (exact_only(s, name)) {
            if (!held(s, name)) best_sample(s, name, NULL);
            printf("hill: %s: candidate\n", name);
            take(s, name, ctree, b);
            int moved = 0;
            for (size_t r = 0; r < rewards->len; r++)
                if (of(s, r, name)) moved |= !num_eq(b[r].v[0], B.rings[r].k[B.rings[r].n - 1].v);
            if (moved) printf("hill: %s: candidate again\n", name), take(s, name, ctree, b);
        } else if (jtext(jget(inst, "exclusive")) && jtext(jget(inst, "serve"))) {
            while (held(s, name) < 3) best_sample(s, name, NULL);
            for (size_t r = 0; r < rewards->len; r++)
                if (of(s, r, name)) a[r] = ring_vals(&B.rings[r]);
            printf("hill: %s: the candidate alone, %lld runs against the best's last %zu\n", name, 2 * blocks, held(s, name));
            for (long long k = 0; k < 2 * blocks; k++) take(s, name, ctree, b);
        } else {
            for (long long k = 0; k < blocks; k++) {
                printf("hill: %s, block %lld of %lld: %s\n", name, k + 1, blocks, k % 2 ? "candidate, best, best, candidate" : "best, candidate, candidate, best");
                for (int i = 0; i < 4; i++)
                    if ((i == 0 || i == 3) == !(k % 2)) best_sample(s, name, a);
                    else take(s, name, ctree, b);
            }
        }
        for (size_t r = 0; r < rewards->len; r++)
            if (of(s, r, name) && (exact_only(s, name) || jtruthy(jget(rewards->v[r], "exact")))) a[r] = ring_vals(&B.rings[r]);
        jset(bdet, name, jget(B.details, name) ? jget(B.details, name) : jnew(JNULL));
        jset(cdet, name, cand_detail ? cand_detail : jnew(JNULL));
        int was = losses;
        judge_rows(s, name, a, b, rows, &wins, &losses);
        for (size_t r = 0; r < rewards->len; r++)
            if (of(s, r, name)) cv[r] = b[r];
        if (losses > was) printf("hill: %s: a loss, so the step stops here\n", name);
    }
    const char *verdict = wins && !losses ? "PROMOTE" : losses ? "REJECT" : "TIE";
    J *values = jnew(JOBJ);
    for (size_t r = 0; r < rows->len; r++) jset(values, rows->k[r], jget(rows->v[r], "candidate"));
    J *entry = entry_of("verdict", &m);
    jset(entry, "best_commit", jstr(head));
    jset(entry, "candidate_commit", jstr(cand));
    jset(entry, "best", jstr(best.product));
    jset(entry, "candidate", jstr(now.product));
    jset(entry, "harness", jstr(best.harness));
    jset(entry, "verdict", jstr(verdict));
    jset(entry, "rows", rows);
    jset(entry, "values", values);
    if (carried->len) jset(entry, "same_for", carried);
    printf("\n%s\n", report(s, rows, verdict, m.label));
    print_detail(s, rows, bdet, cdet);
    if (remote_sha) {
        Buf b = {0};
        dump(&b, entry);
        printf("%s\n", bstr(&b));
        return 0;
    }
    J *all = gather_remotes(losses > 0);
    int promote = !strcmp(verdict, "PROMOTE"), reject = !strcmp(verdict, "REJECT");
    Buf why = {0};
    bprintf(&why, "%s: %s", m.label, verdict);
    for (size_t i = 0; i < all->len; i++) {
        J *e = all->v[i];
        printf("\n%s\n", report(s, jget(e, "rows"), jtext_or(e, "verdict", "TIE"), jtext_or(e, "label", "?")));
        promote |= is(e, "verdict", "PROMOTE");
        reject |= is(e, "verdict", "REJECT");
        bprintf(&why, "\n%s: %s", jtext_or(e, "label", "?"), jtext_or(e, "verdict", "?"));
    }
    append(entry);
    for (size_t i = 0; i < all->len; i++) {
        J *e = jnew(JOBJ);
        for (size_t k = 0; k < all->v[i]->len; k++)
            if (strcmp(all->v[i]->k[k], "at")) jset(e, all->v[i]->k[k], all->v[i]->v[k]);
        append(e);
    }
    for (size_t r = 0; r < rows->len; r++) {
        J *row = rows->v[r];
        if (!is(row, "result", "tie")) {
            char *fa, *fc;
            pair_fmt(s, rows->k[r], jget(row, "best"), jget(row, "candidate"), &fa, &fc);
            bprintf(&why, "\n  %s: %s -> %s (%s)", name_of(s, rows->k[r]), fa, fc, jtext(jget(row, "result")));
        }
    }
    if (promote && !reject) {
        char *c = land(head, tree, message);
        for (size_t i = 0; i < nworkers; i++)
            if (workers[i].pid && !strcmp(workers[i].tree, B.tree)) worker_kill(&workers[i]);
        B.commit = c;
        B.tree = ctree;
        for (size_t r = 0; r < rewards->len; r++)
            if (cv[r].n) {
                B.rings[r].n = 0;
                for (size_t i = 0; i < cv[r].n; i++) ring_push(&B.rings[r], cv[r].v[i]);
            }
        for (size_t i = 0; i < fps->len; i++) jset(B.fps, fps->k[i], fps->v[i]);
        for (size_t i = 0; i < cdet->len; i++) jset(B.details, cdet->k[i], cdet->v[i]);
        printf("\nhill: PROMOTE: landed %.12s, the new best\n", c);
        return 0;
    }
    char *ref = shelve(head, tree, message, bstr(&why));
    printf("\nhill: %s: kept as %s\n", reject ? "REJECT" : "TIE", ref);
    return 1;
}

static int listen_fd = -1;
static char *version;

static void end_step(void) {
    unsetenv("GIT_INDEX_FILE");
    gather_remotes(1);
    for (size_t i = 0; i < nworkers; i++)
        if (workers[i].pid && (!B.tree || strcmp(workers[i].tree, B.tree))) worker_kill(&workers[i]);
}

static void quit_judge(void) {
    for (size_t i = 0; i < nworkers; i++) worker_kill(&workers[i]);
    _exit(0);
}

static void serve_one(int c) {
    Buf got = {0};
    char *line;
    if (read_line(c, &got, now_s() + 5, &line) <= 0) return;
    J *req = parse(line, strlen(line));
    if (!req || req->t != JOBJ) return;
    if (!is(req, "version", version) || is(req, "cmd", "stop")) {
        close(listen_fd);
        dprintf(c, "\x1erestart\n");
        quit_judge();
    }
    fflush(stdout);
    fflush(stderr);
    int o1 = dup(1), o2 = dup(2);
    dup2(c, 1);
    dup2(c, 2);
    volatile int code = 2;
    if (setjmp(stop_env)) {
        fflush(stdout);
        fprintf(stderr, "%s: %s\n", stop_kind, stop_text);
    } else
        caller_fd = c, code = step(jtext(jget(req, "message")), jtext(jget(req, "sha")));
    caller_fd = -1;
    if (!setjmp(stop_env)) end_step();
    fflush(stdout);
    fflush(stderr);
    dprintf(1, "\x1e%d\n", code);
    dup2(o1, 1);
    dup2(o2, 2);
    close(o1);
    close(o2);
}

static void idle_work(void) {
    if (!B.commit) return;
    Spec *s = &B.s;
    Strs names = instrument_names(s);
    double t = now_s();
    for (size_t i = 0; i < names.n; i++) {
        size_t any = 0;
        for (size_t r = 0; r < jget(s->spec, "rewards")->len; r++) any |= (size_t)of(s, r, names.v[i]);
        if (!any || B.next[i] > t) continue;
        if (setjmp(stop_env)) {
            fflush(stdout);
            fprintf(stderr, "hill: idle %s: %s: %s\n", names.v[i], stop_kind, stop_text);
            B.next[i] = now_s() + 60;
            return;
        }
        J *inst = jget(jget(s->spec, "instruments"), names.v[i]);
        int alone = jtext(jget(inst, "exclusive")) != NULL;
        size_t want = exact_only(s, names.v[i]) ? 1 : alone ? 3 : RING;
        if (held(s, names.v[i]) < want) best_sample(s, names.v[i], NULL);
        else if (!alone && !exact_only(s, names.v[i])) best_sample(s, names.v[i], NULL);
        B.next[i] = now_s() + (held(s, names.v[i]) < want ? 0 : alone || exact_only(s, names.v[i]) ? 1e9 : REFRESH_S);
        return;
    }
}

static int idle_ms(void) {
    if (!B.commit) return -1;
    Spec *s = &B.s;
    Strs names = instrument_names(s);
    double soonest = -1, t = now_s();
    for (size_t i = 0; i < names.n; i++) {
        size_t any = 0;
        for (size_t r = 0; r < jget(s->spec, "rewards")->len; r++) any |= (size_t)of(s, r, names.v[i]);
        if (any && (soonest < 0 || B.next[i] < soonest)) soonest = B.next[i];
    }
    if (soonest < 0) return -1;
    return soonest <= t ? 0 : (int)((soonest - t) * 1000) + 1;
}

static void judge_loop(void) {
    signal(SIGPIPE, SIG_IGN);
    setvbuf(stdout, NULL, _IOLBF, 0);
    for (;;) {
        struct pollfd p = {listen_fd, POLLIN, 0};
        int r = poll(&p, 1, idle_ms());
        if (r < 0 && errno == EINTR) continue;
        if (r > 0) {
            int c = accept(listen_fd, NULL, NULL);
            if (c < 0) continue;
            fcntl(c, F_SETFD, FD_CLOEXEC);
            serve_one(c);
            close(c);
        } else if (!r)
            idle_work();
    }
}

static struct sockaddr_un judge_addr(void) {
    struct sockaddr_un a = {0};
    a.sun_family = AF_UNIX;
    strcpy(a.sun_path, ".hill/run/judge.sock");
    return a;
}

static int connect_judge(void) {
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un a = judge_addr();
    if (fd < 0 || connect(fd, (struct sockaddr *)&a, sizeof a)) {
        if (fd >= 0) close(fd);
        return -1;
    }
    fcntl(fd, F_SETFD, FD_CLOEXEC);
    return fd;
}

static int start_judge(int foreground) {
    hill_dir();
    int start = open(run_path("judge.start"), O_RDWR | O_CREAT | O_CLOEXEC, 0666);
    if (start < 0 || flock(start, LOCK_EX)) stop("ERROR", "cannot lock %s", run_path("judge.start"));
    int fd = connect_judge();
    if (fd >= 0) {
        close(start);
        if (foreground) stop("REFUSE", "a judge already runs here");
        return fd;
    }
    int life = open(run_path("judge.pid"), O_RDWR | O_CREAT | O_CLOEXEC, 0666);
    if (life < 0 || flock(life, LOCK_EX)) stop("ERROR", "cannot lock %s", run_path("judge.pid"));
    unlink(".hill/run/judge.sock");
    listen_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un a = judge_addr();
    if (listen_fd < 0 || bind(listen_fd, (struct sockaddr *)&a, sizeof a) || listen(listen_fd, 16)) stop("ERROR", "cannot listen on .hill/run/judge.sock: %s", strerror(errno));
    fcntl(listen_fd, F_SETFD, FD_CLOEXEC);
    if (foreground) {
        close(start);
        dprintf(life, "%d\n", (int)getpid());
        judge_loop();
    }
    fflush(stdout);
    fflush(stderr);
    pid_t pid = fork();
    if (pid < 0) stop("ERROR", "fork: %s", strerror(errno));
    if (!pid) {
        setsid();
        close(start);
        int nul = open("/dev/null", O_RDONLY), log = open(fmtstr("%s/judge.log", logs_dir()), O_WRONLY | O_CREAT | O_APPEND, 0666);
        dup2(nul, 0);
        dup2(log, 1);
        dup2(log, 2);
        close(nul);
        close(log);
        if (ftruncate(life, 0) == 0) dprintf(life, "%d\n", (int)getpid());
        judge_loop();
    }
    close(listen_fd);
    close(life);
    close(start);
    fd = connect_judge();
    if (fd < 0) stop("ERROR", "the judge did not start (see %s/judge.log)", logs_dir());
    return fd;
}

static int cmd_step(const char *message, const char *sha) {
    J *req = jnew(JOBJ);
    jset(req, "version", jstr(version));
    jset(req, "cmd", jstr(sha ? "remote" : "step"));
    if (message) jset(req, "message", jstr(message));
    if (sha) jset(req, "sha", jstr(sha));
    Buf line = {0};
    dump(&line, req);
    bputc(&line, '\n');
    signal(SIGPIPE, SIG_IGN);
    for (int tries = 0; tries < 2; tries++) {
        int fd = start_judge(0);
        if (write(fd, line.p, line.n) != (ssize_t)line.n) stop("ERROR", "the judge closed the connection");
        Buf got = {0};
        char chunk[65536];
        size_t shown = 0;
        ssize_t k;
        while ((k = read(fd, chunk, sizeof chunk)) > 0 || (k < 0 && errno == EINTR)) {
            if (k < 0) continue;
            size_t from = got.n;
            bput(&got, chunk, (size_t)k);
            char *mark = memchr(got.p + from, 0x1e, got.n - from);
            size_t upto = mark ? (size_t)(mark - got.p) : got.n;
            if (upto > shown) fwrite(got.p + shown, 1, upto - shown, stdout), fflush(stdout), shown = upto;
        }
        close(fd);
        bstr(&got);
        char *mark = memchr(got.p, 0x1e, got.n);
        if (!mark) stop("ERROR", "the judge stopped mid-step (see %s/judge.log)", logs_dir());
        if (strncmp(mark + 1, "restart", 7)) return atoi(mark + 1);
    }
    stop("ERROR", "the judge keeps asking for a restart");
    return 2;
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
        if (kept && n && jget(oi->v[i], "budget_s") && budget_of(n) > budget_of(oi->v[i])) spush(found, fmtstr("%s was given a longer budget", oi->k[i]));
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
            int promoted = 0;
            for (size_t i = 0; i < l.n; i++) {
                J *e = l.v[i];
                if (!is(e, "type", "verdict") || !is(e, "best", best.product) || !is(e, "candidate", cand.product) || !is(e, "harness", best.harness)) continue;
                promoted += is(e, "verdict", "PROMOTE");
                if (is(e, "verdict", "REJECT")) spush(&problems, fmtstr("%s judged this change REJECT", jtext_or(e, "label", "a machine")));
            }
            if (!promoted) spush(&problems, "no verdict lands this change: run hill step");
        }
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
              (is(e, "type", "verdict") && (is(e, "verdict", "PROMOTE") || is(e, "verdict", "TIE")) && is(e, "candidate", d.product))))
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
    int ok = argc == 2 && (!strcmp(argv[1], "table") || !strcmp(argv[1], "check") || !strcmp(argv[1], "hook") || !strcmp(argv[1], "judge") || !strcmp(argv[1], "stop"));
    ok |= argc == 3 && !strcmp(argv[1], "step");
    ok |= argc == 4 && !strcmp(argv[1], "step") && !strcmp(argv[2], "--remote");
    if (!ok) {
        fprintf(stderr, "usage: hill step \"<message>\" | table | check | hook | judge | stop\n");
        return 2;
    }
    if (setjmp(stop_env)) {
        fflush(stdout);
        fprintf(stderr, "%s: %s\n", stop_kind, stop_text);
        return 2;
    }
    root_dir = find_root();
    if (chdir(root_dir)) stop("ERROR", "cannot enter %s", root_dir);
    struct stat self;
    version = !stat(argv[0], &self) ? fmtstr("%lld.%lld.%lld", (long long)self.st_ino, (long long)self.st_mtime, (long long)self.st_size) : "?";
    if (!strcmp(argv[1], "check")) return cmd_check();
    if (!strcmp(argv[1], "hook")) return cmd_hook();
    if (!strcmp(argv[1], "judge")) return start_judge(1), 0;
    if (!strcmp(argv[1], "stop")) {
        int fd = connect_judge();
        if (fd >= 0 && write(fd, "{\"cmd\": \"stop\"}\n", 16) == 16) {
            char c;
            while (read(fd, &c, 1) > 0) {}
        }
        return 0;
    }
    if (!strcmp(argv[1], "step")) return argc == 4 ? cmd_step(NULL, argv[3]) : cmd_step(argv[2], NULL);
    Spec s = load_spec_file();
    return cmd_table(&s);
}

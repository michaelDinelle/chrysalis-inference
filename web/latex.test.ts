// LaTeX documents as the chat shows them (latex.ts), against what LaTeX itself makes of them: the numbers it gives
// sections, equations, tables, figures, theorems and references, what \ref and \cite then print, and each block's
// place in the source, which pointing and retyping use.
import { expect, test } from "bun:test";
import { latexToHtml } from "./latex";

// The document's text as read, without KaTeX's MathML copy: blocks apart, a number in a heading apart from its title
const textOf = (html: string) => html.replace(/<span class="katex-mathml">[\s\S]*?<\/math><\/span>/g, "")
  .replace(/<\/?(p|div|h\d|li|td|tr|table|figcaption|header|section|article|nav|figure|dt|dd|ol|ul|blockquote)\b[^>]*>|<\/span>(?=[^<]*<\/h\d)/g, " ")
  .replace(/<[^>]+>/g, "").replace(/&#(\d+);/g, (_m, n) => String.fromCharCode(Number(n))).replace(/[\s\u200b]+/g, " ").trim();

const PAPER = String.raw`\documentclass{article}
\usepackage{amsmath}
\newtheorem{thm}{Theorem}
\newtheorem{lem}{Lemma}
\title{Numbers}
\author{A \and B}
\begin{document}
\maketitle
\section{One}\label{sec:one}
First $x$.
\begin{equation}\label{eq:a}
a = b
\end{equation}
\begin{equation*}
c = d
\end{equation*}
\subsection{One point one}\label{sec:oo}
\begin{align}
e &= f \label{eq:b} \\
g &= h \nonumber \\
i &= j \label{eq:c}
\end{align}
\section{Two}
\begin{table}[h]\caption{T}\label{tab:t}\begin{tabular}{lr}a & 1 \\ b & 2\end{tabular}\end{table}
\begin{figure}\includegraphics{x.png}\caption{F}\label{fig:f}\end{figure}
\begin{thm}\label{thm:t}True.\end{thm}
\begin{lem}Also.\end{lem}
\begin{thm}Again.\end{thm}
See \eqref{eq:a}, \eqref{eq:b}, \ref{eq:c}, \ref{sec:one}, \ref{sec:oo}, \ref{tab:t}, \ref{fig:f}, \ref{thm:t} and \ref{nowhere}.
Cited \cite{beta} then \cite{alpha,beta}.
\begin{thebibliography}{9}
\bibitem{alpha} Alpha.
\bibitem{beta} Beta.
\end{thebibliography}
\end{document}`;

test("numbers as LaTeX gives them: sections, equations (a row with \\nonumber and a starred display take none), tables, figures and each theorem's own count; \\ref and \\eqref print them, an unknown label ??, \\cite the bibliography's order", () => {
  const text = textOf(latexToHtml(PAPER));
  expect(text).toContain("1 One");
  expect(text).toContain("1.1 One point one");
  expect(text).toContain("2 Two");
  expect([...text.matchAll(/\((\d)\)/g)].map((m) => m[1]).slice(0, 3)).toEqual(["1", "2", "3"]);
  expect(text).toContain("Table 1: T");
  expect(text).toContain("Figure 1: F");
  expect([...text.matchAll(/(Theorem|Lemma) (\d)\./g)].map((m) => m[0])).toEqual(["Theorem 1.", "Lemma 1.", "Theorem 2."]);
  expect(text).toContain("See (1), (2), 3, 1, 1.1, 1, 1, 1 and ??.");
  expect(text).toContain("Cited [2] then [1, 2].");
  expect(text).toMatch(/Numbers A B/);
});

test("each block's place in the source is that block's source: a heading, a paragraph, a display, a list item, a table", () => {
  const src = String.raw`\section{Intro}
Some \emph{text} here.

\begin{equation}
x = 1
\end{equation}
\begin{itemize}
  \item First item
  \item Second item
\end{itemize}
\begin{tabular}{ll}
a & b
\end{tabular}`;
  const html = latexToHtml(src), parts = [...html.matchAll(/data-at="(\d+)" data-end="(\d+)"/g)].map((m) => src.slice(Number(m[1]), Number(m[2])));
  expect(parts).toEqual(["\\section{Intro}", "Some \\emph{text} here.", "\\begin{equation}\nx = 1\n\\end{equation}",
    "\\begin{itemize}\n  \\item First item\n  \\item Second item\n\\end{itemize}", "\\item First item", "First item", "\\item Second item", "Second item",
    "\\begin{tabular}{ll}\na & b\n\\end{tabular}"]);
});

test("what is not LaTeX's stays text: markup in it is shown, not made, a link only goes to the web or mail", () => {
  const html = latexToHtml(String.raw`Hi <script>alert(1)</script> <img src=x onerror=alert(1)>
\href{javascript:alert(1)}{bad} \href{https://example.com/a?b=1}{good} \url{mailto:a@b.c}`);
  expect(html).not.toMatch(/<script|<img|javascript:/i);
  expect(textOf(html)).toContain("Hi <script>alert(1)</script> <img src=x onerror=alert(1)>");
  expect([...html.matchAll(/href="([^"]*)"/g)].map((m) => m[1])).toEqual(["https://example.com/a?b=1", "mailto:a@b.c"]);
});

test("environments and commands it does not know show their text, and nothing stops the document", () => {
  const text = textOf(latexToHtml(String.raw`\begin{weird}[opt]{arg} inside \foo{bar} and \baz \end{weird}
\begin{tikzpicture}\draw (0,0) -- (1,1);\end{tikzpicture}
Unclosed \textbf{bold
\begin{itemize} \item never closed`));
  for (const words of ["inside bar and", "tikzpicture picture (not drawn here)", "Unclosed bold", "never closed"]) expect(text).toContain(words);
});

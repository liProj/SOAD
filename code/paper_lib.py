"""LaTeX helpers. Prose uses <<key>> markers filled from results; fill() fails on any marker left over."""
import re


def f2(x): return f"{float(x):.2f}"
def f3(x): return f"{float(x):.3f}"
def f1(x): return f"{float(x):.1f}"
def sg(x, n=3): return f"{float(x):+.{n}f}".replace("-", "$-$")
def pct(x, n=1): return f"{100 * float(x):.{n}f}\\%"
def esc(s): return str(s).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")


def tabular(rows, header, align, caption, label, note=None, size="\\small"):
    out = ["\\begin{table}[!ht]", "\\centering", size, f"\\caption{{{caption}}}", f"\\label{{{label}}}",
           f"\\begin{{tabular}}{{{align}}}", "\\toprule", (header if isinstance(header, str) else " & ".join(header) + " \\\\"), "\\midrule"]
    for r in rows:
        out.append(r if isinstance(r, str) else " & ".join(str(c) for c in r) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    if note:
        out.append(f"\\par\\vspace{{2pt}}\\parbox{{0.97\\linewidth}}{{\\footnotesize {note}}}")
    out.append("\\end{table}")
    return "\n".join(out)


def figure(name, caption, label, width="\\linewidth"):
    return (f"\\begin{{figure}}[!ht]\n\\centering\n\\includegraphics[width={width}]{{../figures/{name}.pdf}}\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{figure}}")


def fill(text, N):
    def rep(m):
        k = m.group(1)
        if k not in N:
            raise KeyError(f"marker <<{k}>> has no value")
        return str(N[k])
    out = re.sub(r"<<([A-Za-z0-9_]+)>>", rep, text)
    left = re.findall(r"<<[^>]*>>", out)
    if left:
        raise ValueError(f"unfilled markers: {left[:5]}")
    return out

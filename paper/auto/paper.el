(TeX-add-style-hook
 "paper"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "12pt" "draft")))
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art12"))
 :latex)


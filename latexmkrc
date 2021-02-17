$clean_ext = 'run.xml tex.bak bbl bcf fdb_latexmk run tdo blg bcf aux snm out toc nav vrb fls log tex.backup';
$pdf_mode = 1;
$pdflatex = 'pdflatex -shell-escape -interaction=nonstopmode %O %S -file-line-error -synctex=1';
$cleanup_includes_cusdep_generated = 1;


add_cus_dep('svg', 'pdf', 0, 'svg2pdf');
sub svg2pdf {
        return system("inkscape --export-area-page --export-filename=\"$_[0].pdf\" \"$_[0].svg\"");
}

add_cus_dep('py', 'pdf', 0, 'py2pdf');
sub py2pdf {
        rdb_ensure_file($rule, "code/common.py");
        rdb_ensure_file($rule, "code/matplotlibrc");
        return system("python3 \"$_[0].py\" \"$_[0].pdf\"");
}

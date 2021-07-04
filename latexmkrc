$clean_ext = 'run.xml tex.bak bbl bcf fdb_latexmk run tdo blg bcf aux snm out toc nav vrb fls log glo gls glg acn acr alg ist tex.backup xmpi';
$pdf_mode = 1;
$pdflatex = 'pdflatex -shell-escape -interaction=nonstopmode %O %S -file-line-error';
$cleanup_includes_cusdep_generated = 1;

push @extra_pdflatex_options, '-synctex=1' ;

add_cus_dep('svg', 'pdf', 0, 'svg2pdf');
sub svg2pdf {
        return system("inkscape --export-area-page --export-filename=\"$_[0].pdf\" \"$_[0].svg\"");
}

add_cus_dep('py', 'pdf', 0, 'py2pdf');
sub py2pdf {
        rdb_ensure_file($rule, "media/generate_figure.py");
        rdb_ensure_file($rule, "media/matplotlibrc");
        return system("python3 media/generate_figure.py --script \"$_[0].py\" --target \"$_[0].pdf\"");
}

add_cus_dep('glo', 'gls', 0, 'makeglo2gls');
add_cus_dep('acn', 'acr', 0, 'makeacn2acr');
sub makeglo2gls {
    system("makeindex -s '$_[0]'.ist -t '$_[0]'.glg -o '$_[0]'.gls '$_[0]'.glo");
}
sub makeacn2acr {
    system("makeindex -s '$_[0]'.ist -t '$_[0]'.alg -o '$_[0]'.acr '$_[0]'.acn");
}


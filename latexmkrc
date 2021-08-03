$clean_ext = 'run.xml tex.bak bbl bcf fdb_latexmk run tdo blg bcf aux snm out toc nav vrb fls log glo gls glg acn acr alg ist tex.backup xmpi';
$recorder = 1;
$pdf_mode = 1;
#$pdflatex = 'pdflatex -fmt astoeckel_phd_thesis_2021 -shell-escape -interaction=nonstopmode %O %S -file-line-error';
$pdflatex = 'pdflatex -shell-escape -interaction=nonstopmode %O %S -file-line-error';
$cleanup_includes_cusdep_generated = 1;

push @extra_pdflatex_options, '-synctex=1' ;

# See https://tex.stackexchange.com/a/37730
#add_cus_dep('ltx', 'fmt', 1, 'compilepreamble');
sub compilepreamble {
    print "Preamble compiling for '$_[0]'...\n";
    my $fls_file = "$_[0].fls";
    my $source = "$_[0].ltx";
    my $fmt_file = "$_[0].fmt";
    my $return = system( "pdflatex", "-interaction=batchmode",
                         "-ini", "-recorder", "-jobname=$_[0]",
                         "&pdflatex $source \\dump" );
    if ($return) {
        warn "Error in making format file '$fmt_file'\n";
       return $return;
    }
    my %input_files = ();
    my %output_files = ();
    $return = parse_fls( $fls_file, \%input_files, \%output_files );
    if ($return) {
        warn "No fls file '$fls_file' made; I cannot get dependency data\n";
        return 0;
    }
    # Use latexmk's internal variables and subroutines for setting the
    #   dependency information.
    # Note that when this subroutine is called to implement a custom
    #   dependency, the following variables are set:
    #       $rule  contains the name of the current rule (as in the
    #              fdb_latexmk file)
    #       $PHsource is a pointer to a hash of source file
    #              information for the rule.  The keys of the hash are
    #              the names of the source files of the rule.
    foreach my $file (keys %input_files) {
        rdb_ensure_file( $rule, $file );
    }
    foreach my $file (keys %$PHsource) {
        if ( ! exists $input_files{$file} ) {
            print "   Source file of previous run '$file' ",
                  "IS NO LONGER IN USE\n";
            rdb_remove_files( $rule, $file );
        }
    }
    return 0;
};


add_cus_dep('svg', 'pdf', 0, 'svg2pdf');
sub svg2pdf {
        return system("inkscape --export-area-page --export-filename=\"$_[0].pdf\" \"$_[0].svg\"");
}

add_cus_dep('py', 'pdf', 0, 'py2pdf');
sub py2pdf {
        rdb_ensure_file($rule, "scripts/generate_figure.py");
        rdb_ensure_file($rule, "media/matplotlibrc");
        return system("python3 scripts/generate_figure.py --datafile_cache .datafile_cache.json --script \"$_[0].py\" --target \"$_[0].pdf\"");
}

add_cus_dep('glo', 'gls', 0, 'makeglo2gls');
add_cus_dep('acn', 'acr', 0, 'makeacn2acr');
sub makeglo2gls {
    system("makeindex -s '$_[0]'.ist -t '$_[0]'.glg -o '$_[0]'.gls '$_[0]'.glo");
}
sub makeacn2acr {
    system("makeindex -s '$_[0]'.ist -t '$_[0]'.alg -o '$_[0]'.acr '$_[0]'.acn");
}


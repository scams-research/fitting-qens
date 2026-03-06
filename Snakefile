autocorr_names = ["model_nogauss_oneexp", 'model_nogauss_twoexp', 'model_nogauss_oneexpgauss', 'model_nogauss_twoexpgauss', 'model_gauss_oneexp', 'model_gauss_twoexp', 'model_gauss_oneexpgauss', 'model_gauss_twoexpgauss']

rule:
    conda: "environment.yml"
    name:
        "autocorrelation_evidence"
    input: 
        [f"src/data/{i}.pkl" for i in autocorr_names],
        "src/scripts/si_raf_chain.py"
    output:
        [f"src/tex/output/{i}_evidence.txt" for i in autocorr_names],
        "src/tex/output/best_raf_evidence.txt"
    script:
        "src/scripts/si_raf_chain.py"

rule:
    conda: "environment.yml"
    name:
        "main_text_outputs"
    input: 
        "src/data/kinisi_D",
        "src/data/model_gauss_twoexp.pkl",
        "src/data/rotation_only_model.pkl",
        "src/data/rotation_only_model_iso.pkl",
        "src/data/pLET_aniso_model.pkl",
        "src/data/pLET_iso_model.pkl",
        "src/scripts/main_text_output.py"
    output:
        "src/tex/output/kinisi_D.txt",
        "src/tex/output/D_rot_diff.txt",
        "src/tex/output/D_rot_ratio.txt",
        "src/tex/output/aniso_rot_only_evidence.txt",
        "src/tex/output/iso_rot_only_evidence.txt",
        "src/tex/output/D_fick_let.txt",
        "src/tex/output/D_ratio_let.txt",
        "src/tex/output/evidence_aniso_let.txt",
        "src/tex/output/evidence_iso_let.txt"
    script:
        "src/scripts/main_text_output.py"

rule:
    conda: "environment.yml"
    name:
        "si_text_outputs"
    input: 
        "src/data/lit_approach.pkl",
        "src/scripts/si_text_output.py",
    output:
        "src/tex/output/lit_approach_evidence.txt",
        "src/tex/output/lit_approach_Ds.txt",
        "src/tex/output/lit_approach_Dt.txt",
        "src/tex/output/lit_approach_t0.txt",
        "src/tex/output/lit_approach_t90.txt"
    script:
        "src/scripts/si_text_output.py"

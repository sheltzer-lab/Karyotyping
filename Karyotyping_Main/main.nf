#!/usr/bin/env nextflow

import groovy.json.JsonOutput

// Sample sheet (TSV file)
// Columns: name, highlight, txt
//   name: Name of the sample to be applied as the title of graph (string)
//   R1: Reads 1
//   R2: Reads 2
//
// Example:
// name    R1                   R1
// Sample1 Sample1_R1.fastq.gz  Sample1_R2.fastq.gz
// Sample2 Sample2_R1.fastq.gz  Sample2_R2.fastq.gz
// ...
params.sampleSheet = null
params.bamFiles = null
params.plottingFiles = null

// Reference genome
params.reference = "/oak/stanford/groups/sheltzer/shared/ref/Homo_sapiens_assembly38.fasta"

// UMAP of same size as reads
params.umap = "/oak/stanford/groups/sheltzer/shared/ref/k150.GRCh38.Umap.bed"

// Segmental duplication
params.segdup = "/oak/stanford/groups/sheltzer/shared/ref/hg38-segdup.bed"

// Intervals
params.intervals = "/oak/stanford/groups/sheltzer/shared/ref/wgs_calling_regions.hg38.interval_list"

// Exclude
params.exclude = "/oak/stanford/groups/sheltzer/shared/ref/hg38-centromeres.bed"

// Exclude telomeres+centromeres
params.exclude_all = "/oak/stanford/groups/sheltzer/shared/ref/access-excludes.bed"

// Annotation file
params.annotate = "/oak/stanford/groups/sheltzer/shared/ref/refFlat.txt"

// The vizualization extensions to produce (comma-separated list)
params.exts = "png,pdf"

// The bin sizes to use
params.bins = "50000,70000,80000,85000,90000,95000"

// ---------- params you care about ----------
params.plot_inputs    = null      // TSV => replot mode

// helper: deterministic empty JSON file (stable = good caching)
def ensureJsonFile = { pathLike, stem ->
    if (pathLike && pathLike.toString().trim() !in ['-', '', 'None', 'none', 'NULL', 'null'])
        return file(pathLike)
    def f = file("${stem}.empty.json")
    if (!f.exists()) f.text = "{}"
    return f
}

params.arms_summary = "arms_summary.txt"

//
params.base = 2          // or whatever you want
params.y_log = false     // "true"/"false" as strings or booleans coerced to strings


workflow {
    final boolean REPLOT = params.plot_inputs as boolean

    // -------------------------------
    // REPLOT MODE: only here we honor highlight/correction
    // -------------------------------

    // ---------- Arms summary -> ranges dict ----------
    def arms_data = Channel
        .fromPath(params.arms_summary, checkIfExists: true)
        .splitCsv(header: true, sep: '\t')
        .map { row ->
            def ranges = [:]
            def chrom = (row.chrom as String)              // e.g., 'chr8'
            def arm   = (row.arm   as String)              // 'p' | 'q' | 'NA'
            if (arm != 'NA' && chrom) {
                def key = "${chrom}${arm}"                 // e.g., 'chr8q'
                ranges[key] = [
                    chrom,
                    (row.start_pos as String).toInteger(),
                    (row.end_pos   as String).toInteger()
                ]
            }
            ranges
        }
        .reduce([:]) { acc, ranges -> acc + ranges }       // merge per-row maps

    if (REPLOT) {
        log.info "[Replot] Using TSV: ${params.plot_inputs}"

        def plotting_input = Channel
            .fromPath(params.plot_inputs, checkIfExists: true)
            .splitCsv(header: true, sep: '\t')
            .combine(arms_data)                                 // 'ranges' is a Map like ['chr8q': ['chr8', start, end], ...]
            .map { row, ranges ->
                if (!row.containsKey('cleanname') || !row.containsKey('intersect_cnr') || !row.containsKey('sorted_cns')) {
                    throw new IllegalArgumentException(
                        "plot_inputs TSV must have: cleanname, intersect_cnr, sorted_cns (optional: Highlight_Arm, correction_json)"
                    )
                }

                def cleanname     = row.cleanname as String
                def intersect_cnr = file(row.intersect_cnr)
                def sorted_cns    = file(row.sorted_cns)

                // --- YOUR highlight builder (tokens like "8p, 8q"; no 'chr' prefix in TSV) ---
                def highlight_dict = [:].withDefault { [] as List<List<Integer>> }
                if (row.Highlight_Arm) {
                    row.Highlight_Arm
                        .split(',')
                        .collect { it.trim() }
                        .findAll { it }
                        .each { arm ->
                            def m = arm =~ /(\d+)([pq])/    // e.g., 8q, 17p, X
                            if (m.find()) {
                                def chrom          = m.group(1)           // e.g., '8'
                                def armDesignation = m.group(2)           // 'p' or 'q'
                                def key = "chr${chrom}${armDesignation}"  // e.g., 'chr8q'
                                def entry = ranges[key]
                                if (entry) {
                                    def (chr, start, end) = entry
                                    highlight_dict[chr] << [ (start as Integer), (end as Integer) ]
                                } else {
                                    log.warn "[Replot] No range for '${arm}' -> key '${key}' in arms summary"
                                }
                            } else {
                                log.warn "[Replot] Bad Highlight_Arm token '${arm}' (use like 8p, 17q, X)"
                            }
                        }
                }

                def highlight_json = file("${workflow.workDir}/${cleanname}.highlight.json")
                highlight_json.text = groovy.json.JsonOutput.toJson(highlight_dict)

                // Correction stays a path as before
                def correction_json = ensureJsonFile(
                    row.containsKey('correction_json') && row.correction_json ? row.correction_json : null,
                    "${cleanname}.correction"
                )

                tuple(cleanname, intersect_cnr, sorted_cns, highlight_json, correction_json)
            }

        PLOTTING(plotting_input)
        return
    }




    // ---------- Samples & highlight JSON text per sample ----------
    // def samples = Channel
    //     .fromPath(params.sampleSheet, checkIfExists: true)
    //     .splitCsv(header: true, sep: '\t')
    //     .combine(arms_data)
    //     .map { row, ranges ->
    //         def highlight_dict = [:]
    //         if (row.Highlight_Arm) {
    //             row.Highlight_Arm.split(',').collect { it.trim() }.each { arm ->
    //                 def m = arm =~ /(\d+)([pq])/
    //                 if (m.find()) {
    //                     def chrom = m.group(1)
    //                     def armDesignation = m.group(2)
    //                     def key = "chr${chrom}${armDesignation}"
    //                     def entry = ranges[key]
    //                     if (entry) {
    //                         def (chr, start, end) = entry
    //                         if (!highlight_dict[chr]) highlight_dict[chr] = []
    //                         highlight_dict[chr] << [start, end]
    //                     }
    //                 }
    //             }
    //         }
    //         def json_text = params.use_highlight ? JsonOutput.toJson(highlight_dict) : "{}"
    //         tuple(row.name, file(row.R1), file(row.R2), json_text)
    //     }

// ---------- Samples with empty JSON files by default ----------
    def samples = Channel
        .fromPath(params.sampleSheet, checkIfExists: true)
        .splitCsv(header: true, sep: '\t')
        .map { row -> tuple(row.name, file(row.R1), file(row.R2)) }

    def samples_for_clean     = samples.map { name, R1, R2 -> tuple(name, R1, R2) }
    def bins = Channel.fromList(params.bins?.tokenize(','))
    def bam_channel

    // ---------- Reference index bundle (unchanged from yours) ----------
    def reference = Channel
        .fromPath(params.reference, checkIfExists: true)
        .map { ref -> tuple(ref, file("${ref.parent}/*.{amb,ann,bwt,dict,fai,pac,sa}")) }
        .first()

    // ---------- BAMs (either provided or produced) ----------
    if (params.bamFiles) {
        bam_channel = Channel
            .fromPath(params.bamFiles, checkIfExists: true)
            .splitCsv(header: true, sep: '\t')
            .map { row -> tuple(row.name, file(row.bam_path), file(row.bai_path)) }
    } else {
        CLEAN_READS(samples_for_clean)
        ALIGN_READS(CLEAN_READS.out[0], reference)
        bam_channel = ALIGN_READS.out
    }

    // ---------- CNV analysis for each bin ----------
    def analysis_channel = bam_channel.combine(bins).map { name, bam, bai, bin_size ->
        def cleanname = "${name}_bin${bin_size}"
        [cleanname, bam, bai, file(params.exclude_all), file(params.annotate), file(params.reference), bin_size]
    }

    CNV_ANALYSIS(analysis_channel)
    CNR_TO_BED(CNV_ANALYSIS.out[0])
    CNS_TO_BED(CNV_ANALYSIS.out[0])

    def intersect_input = CNR_TO_BED.out.join(CNS_TO_BED.out)
    INTERSECT_BED(intersect_input)

    def default_highlight_json = file("${workflow.workDir}/default_highlight.json")
    def default_correction_json = file("${workflow.workDir}/default_correction.json")
    default_highlight_json.text = "{}"
    default_correction_json.text = "{}"

    def plotting_input = INTERSECT_BED.out
        .join(CNV_ANALYSIS.out[2])     // (cleanname, sorted_cns)
        .map { cleanname, intersect_cnr, sorted_cns ->
            tuple(cleanname, intersect_cnr, sorted_cns, default_highlight_json, default_correction_json)
        }
    PLOTTING(plotting_input)
}


process MAKE_HIGHLIGHT {
        label 'process_low'
        tag "${name}"

        input:
          tuple val(name), val(json_text)

        output:
          tuple val(name), path("${name}_highlight.json")

        /*
          We always create a file:
          - If json_text == "{}", plotting can treat it as “no highlight”.
          - Keeping it as a declared output stabilizes task hashing for -resume.
        */
        script:
        """
        cat > "${name}_highlight.json" <<'EOF'
        ${json_text}
        EOF
        """

        stub:
        """
        echo '{}' > "${name}_highlight.json"
        """
      }

process CLEAN_READS {
    conda 'bioconda::fastp=0.23.4'
    label 'process_low'
    tag "${cleanname}"
    cpus 4
    publishDir "results/cleaning/", pattern: "*_fastp.{html,json}", mode: 'copy', overwrite: true

    input:
      tuple val(name),
            path(R1),
            path(R2)
    
    output:
      tuple val(cleanname),
            path("clean_${R1}"),
            path("clean_${R2}")
      path("${cleanname}_fastp.html")
      path("${cleanname}_fastp.json")

    script:
      cleanname = name.replaceAll('[^a-zA-Z0-9_]+', '_')
      """
      fastp \
        --thread ${task.cpus} \
        --in1 ${R1} \
        --in2 ${R2} \
        --out1 clean_${R1} \
        --out2 clean_${R2}
      
      mv fastp.html ${cleanname}_fastp.html
      mv fastp.json ${cleanname}_fastp.json
      """
    
    stub:
      cleanname = name.replaceAll('[^a-zA-Z0-9_]+', '_')
      """
      touch clean_${R1}
      touch clean_${R2}
      touch ${cleanname}_fastp.html
      touch ${cleanname}_fastp.json
      """
}

process ALIGN_READS {
    conda 'bioconda::bwa=0.7.17 bioconda::samtools=1.17'
    label 'process_medium'
    tag "${cleanname}"
    cpus { 8 * task.attempt }
    errorStrategy 'retry'
    maxRetries 3

    input:
      tuple val(cleanname),
            path(clean_r1),
            path(clean_r2)
      tuple path(reference),
            path(indexes)
    
    output:
      tuple val(cleanname),
            path("${cleanname}.sorted.bam"),
            path("${cleanname}.sorted.bam.bai")

    script:
      """
      bwa mem \
        -t ${task.cpus} \
        -R "@RG\\tID:${cleanname}\\tSM:${cleanname}" \
        ${reference} \
        ${clean_r1} \
        ${clean_r2} > ${cleanname}.bam

      samtools sort \
        --threads ${task.cpus} \
        ${cleanname}.bam \
        -o ${cleanname}.sorted.bam

      samtools index \
        -@ ${task.cpus} \
        ${cleanname}.sorted.bam
      """
    
    stub:
      """
      touch ${cleanname}.sorted.bam
      touch ${cleanname}.sorted.bam.bai
      """
}

process COMPUTE_DEPTH {
    conda 'bioconda::mosdepth=0.3.6'
    label 'process_low'
    tag "${cleanname}"
    publishDir "results/cleaning/", mode: 'copy', overwrite: true

    input:
      tuple val(cleanname),
            path("${cleanname}.sorted.bam"),
            path("${cleanname}.sorted.bam.bai")
    
    output:
      path("${cleanname}.mosdepth.global.dist.txt")
      path("${cleanname}.mosdepth.region.dist.txt")
      path("${cleanname}.mosdepth.summary.txt")
      path("${cleanname}-COVERAGE.html")
      
    script:
      """
      mosdepth -n --fast-mode --by 500 ${cleanname} ${cleanname}.sorted.bam
      wget https://raw.githubusercontent.com/brentp/mosdepth/master/scripts/plot-dist.py
      python plot-dist.py --output ${cleanname}-COVERAGE.html ${cleanname}.mosdepth.global.dist.txt
      """
}

process CNV_ANALYSIS {
    conda "${projectDir}/env/cnvkit.yml" 
    label 'process_medium'
    tag "${cleanname}"
    publishDir "results/cnv/${cleanname}/", mode: 'copy', overwrite: true
    cpus 8
    
    input:
        tuple val(cleanname),
              path(bam),
              path(bai),
              path(exclude_all),
              path(annotate),
              path(ref_fasta),
              val(bin_size)
              
    output:
      tuple val(cleanname),
            path("${cleanname}.cnr"),
            path("${cleanname}.call.cns")
      path("${cleanname}.bintest.cns")
      tuple val(cleanname),
            path("${cleanname}.cns")
      path("${cleanname}-diagram.pdf")
      path("${cleanname}-scatter.png")

    script:
      """
      
      
      cnvkit.py batch ${bam} \
        --target-avg-size ${bin_size} \
        --access ${exclude_all} \
        --output-reference ${cleanname}_reference.cnn \
        -m wgs \
        -f ${ref_fasta} \
        --annotate ${annotate} \
        -n \
        --output-dir . \
        --diagram \
        --scatter \
        -p ${task.cpus}
        
      # Extract base name without extension
      base_name=\$(basename ${bam} .bam)
      
      # Rename all output files to match expected output names
      if [ -f "\${base_name}.cnr" ]; then
        mv "\${base_name}.cnr" "${cleanname}.cnr"
      fi
      
      if [ -f "\${base_name}.cns" ]; then
        mv "\${base_name}.cns" "${cleanname}.cns"
      fi
      
      if [ -f "\${base_name}.bintest.cns" ]; then
        mv "\${base_name}.bintest.cns" "${cleanname}.bintest.cns"
      fi
      
      if [ -f "\${base_name}.call.cns" ]; then
        mv "\${base_name}.call.cns" "${cleanname}.call.cns"
      fi
      
      if [ -f "\${base_name}-diagram.pdf" ]; then
        mv "\${base_name}-diagram.pdf" "${cleanname}-diagram.pdf"
      fi
      
      if [ -f "\${base_name}-scatter.png" ]; then
        mv "\${base_name}-scatter.png" "${cleanname}-scatter.png"
      fi
      
      # List all files to verify
      ls -la
      
      echo "=== CNV_ANALYSIS Complete ==="
      """
   
    stub:
      """
      touch ${cleanname}.cnr
      touch ${cleanname}.cns
      touch ${cleanname}.bintest.cns
      touch ${cleanname}.call.cns
      touch ${cleanname}-diagram.pdf
      touch ${cleanname}-scatter.png
      """
}

process CNR_TO_BED {
    conda "python"
    label 'process_low'
    tag "${cleanname}"
    publishDir "results/bedfiles/${cleanname}/", mode: 'copy', overwrite: true

    input:
      tuple val(cleanname),
            path(cnr),
            path(cns)

    output:
      tuple val(cleanname),
            path("${cleanname}.sorted_ratios.bed")

    script:
      """
      cnr_to_bed.py ${cnr} ${cleanname}.sorted_ratios.bed
      """
    
    stub:
      """
      touch ${cleanname}/${cleanname}.sorted_ratios.bed
      """
}

process CNS_TO_BED {
    conda "python"
    label 'process_low'
    tag "${cleanname}"
    publishDir "results/bedfiles/${cleanname}/", mode: 'copy', overwrite: true

    input:
      tuple val(cleanname),
            path(cnr),
            path(cns)

    output:
      tuple val(cleanname),
            path("${cleanname}.sorted_copynums.bed")

    script:
      """
      cns_to_bed.py ${cns} ${cleanname}.sorted_copynums.bed
      """
    
    stub:
      """
      touch ${cleanname}/${cleanname}.sorted_copynums.bed
      """
}

process INTERSECT_BED {
    conda 'bioconda::bedtools=2.30.0'
    label 'process_low'
    tag "${cleanname}"
    publishDir "results/intersect/${cleanname}", mode: 'copy', overwrite: true

    input:
      tuple val(cleanname),
            path(ratios_bed),
            path(copynums_bed)

    output:
      tuple val(cleanname),
            path("${cleanname}.Intersect.cnr")
            
    script:
      """
      echo -e "chromosome\tstart\tend\tgene\tdepth\tlog2\tweight\tlog2Seg\tcn" > ${cleanname}.Intersect.cnr
      bedtools intersect -wb -a ${ratios_bed} -b ${copynums_bed} | cut -f1-7,12-13 >> ${cleanname}.Intersect.cnr
      """
    
    stub:
      """
      touch ${cleanname}.Intersect.cnr
      """
}

process PLOTTING {
  conda "${projectDir}/env/plotting.yml"
  label 'process_low'
  tag "${cleanname}"
  publishDir "results/plots/${cleanname}/", mode: 'copy', overwrite: true

  input:
    tuple val(cleanname),
          path(intersect_cnr),
          path(sorted_cns),
          path(highlight_json),
          path(correction_json)

  output:
    path("${cleanname}_plot.png")

  script:
  """
  Plotting-nice.py \
    ${intersect_cnr} \
    ${sorted_cns} \
    ${cleanname} \
    ${cleanname}_plot.png \
    ${params.base} \
    ${params.y_log} \
    ${highlight_json} \
    ${correction_json}
  """
}


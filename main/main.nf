#!/usr/bin/env nextflow

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
params.sampleSheet = "Karyotypes.txt"
params.bamFiles = null

// Reference genome
params.reference = "ref/Homo_sapiens_assembly38.fasta"

// UMAP of same size as reads
params.umap = "ref/k150.GRCh38.Umap.bed"

// Segmental duplication
params.segdup = "ref/hg38-segdup.bed"

// Intervals
params.intervals = "ref/wgs_calling_regions.hg38.interval_list"

// Exclude
params.exclude = "ref/hg38-centromeres.bed"

// Exclude telomeres+centromeres
params.exclude_all = "ref/access-excludes.bed"

// Annotation file
params.annotate = "ref/refFlat.txt"

// The vizualization extensions to produce (comma-separated list)
params.exts = "png,pdf"

// The bin sizes to use
params.bins = "100000,50000"

workflow {
    // Define input channels based on whether BAM files or FASTQ files are provided
    def bam_channel
    def samples = Channel.fromPath(params.sampleSheet, checkIfExists: true) \
                | splitCsv(header: true, sep: '\t') \
                | map { row -> tuple(row."name", file("${row.R1}"), file("${row.R2}")) }
    def reference = Channel.fromPath(params.reference, checkIfExists: true) \
                | map { ref -> tuple(ref, file("${ref.parent}/*.{amb,ann,bwt,dict,fai,pac,sa}")) } \
                | first
    def umap = Channel.fromPath(params.umap, checkIfExists: true) \
                | map { u -> tuple(u, file("${u.parent}/*.idx")) } \
                | first
    def segdup = Channel.fromPath(params.segdup, checkIfExists: true)
    def intervals = Channel.fromPath(params.intervals, checkIfExists: true)
    def exclude = Channel.fromPath(params.exclude, checkIfExists: true)
    def exclude_all = Channel.fromPath(params.exclude_all, checkIfExists: true)
    def annotate = Channel.fromPath(params.annotate, checkIfExists: true)
    def ref_fasta = Channel.fromPath(params.reference, checkIfExists: true)
    def exts = Channel.fromList(params.exts?.tokenize(','))
    def bins = Channel.fromList(params.bins?.tokenize(','))
    // Determine whether to use BAM files or FASTQ files
    if (params.bamFiles) {
        // If BAM files are provided, read from the BAM list
        bam_channel = Channel.fromPath(params.bamFiles, checkIfExists: true) \
            | splitCsv(header: true, sep: '\t') \
            | map { row -> tuple(row."name", file(row."bam_path"), file(row."bai_path")) }
    } else {
        // Run CLEAN_READS and ALIGN_READS processes for FASTQ files
        CLEAN_READS(samples)
        ALIGN_READS(CLEAN_READS.out[0], reference)
        bam_channel = ALIGN_READS.out
    }
    // Combine the BAM channel with additional inputs
    def analysis_channel = bam_channel.map { tuple ->
    // Unpack the tuple
      def (name, bam, bai) = tuple 
      return [name, bam, bai, file(params.exclude_all), file(params.annotate), file(params.reference)]
    }
    // Run the analysis processes
    COMPUTE_DEPTH(bam_channel)
    CNV_ANALYSIS(analysis_channel)
    CNR_TO_BED(CNV_ANALYSIS.out[0])
    CNS_TO_BED(CNV_ANALYSIS.out[0])

    def intersect_input = CNR_TO_BED.out.join(CNS_TO_BED.out)
    INTERSECT_BED(intersect_input)

    def plotting_input = INTERSECT_BED.out.join(CNV_ANALYSIS.out[2])
    plotting_input.view { "PLOTTING input: ${it}" }
    PLOTTING(plotting_input)
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
            path("${cleanname}.sorted.bam"),
            path("${cleanname}.sorted.bam.bai"),
            path(exclude_all),
            path(annotate),
            path(ref_fasta)
            
    output:
      tuple val(cleanname),
            path("${cleanname}.sorted.cnr"),
            path("${cleanname}.sorted.call.cns")
      path("${cleanname}.sorted.bintest.cns")
      tuple val(cleanname),
            path("${cleanname}.sorted.cns")
      path("${cleanname}.sorted-diagram.pdf")
      path("${cleanname}.sorted-scatter.png")

    script:
      """
      cnvkit.py batch ${cleanname}.sorted.bam \
        --target-avg-size 100000 \
        --access ${exclude_all} \
        --output-reference New_Ref.cnn \
        -m wgs \
        -f ${ref_fasta} \
        --annotate ${annotate} \
        -n \
        --output-dir .\
        --diagram \
        --scatter \
        -p ${task.cpus}
      """
   
    stub:
      """
      touch ${cleanname}/${cleanname}.sorted.cnr
      touch ${cleanname}/${cleanname}.sorted.cns
      touch ${cleanname}/${cleanname}.sorted.bintest.cns
      touch ${cleanname}/${cleanname}.sorted.call.cns
      touch ${cleanname}/${cleanname}.sorted-diagram.pdf
      touch ${cleanname}/${cleanname}.sorted-scatter.png
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
      echo -e "chromosome\tstart\tend\tgene\tdepth\tlog2\tweight\tcn" > ${cleanname}.Intersect.cnr
      bedtools intersect -wb -a ${ratios_bed} -b ${copynums_bed} | cut -f1-7,13 >> ${cleanname}.Intersect.cnr
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
            path(sorted_cns)

    output:
      path("${cleanname}_plot.png")

    script:
      """
      Plotting.py ${intersect_cnr} ${sorted_cns} ${cleanname} ${cleanname}_plot.png
      """
    
    stub:
      """
      touch ${cleanname}_plot.png
      """
}

Creates the pro-atomic database needed for Hirshfeld-based partitioning.

Since internal consistency with HORTON is crucial, we will use its internal codes. Therefore, two things are assumed:
- the HORTON scripts are accessible in the path;
- a quantum chemistry code (Gaussian, ORCA, CP2Kor Psi4) is accessible in the path.

Copied directly from HORTON manual:

5.1.2. horton-atomdb.py – Build a pro-atom database

The horton-atomdb.py script should be used in three steps to build a pro-atom database:

    Generate input files for isolated atom computations using one of the following programs: Gaussian03/09, Orca, PSI4 or CP2K. The following example generates Gaussian09 input files for hydrogen, carbon, nitrogen and oxygen:

    horton-atomdb.py input g09 1,6-8 template.com

    The template.com file is used to generate the input files and will be discussed in detail below. A series of directories is created with input files for the atomic computations, like 001__h_001_q+00, 001__h_002_q-01, 001__h_003_q-02, etc. Optional arguments can be used to control the range of cations and anions, the spin multiplicities, etc. Also a run_g09.sh script is generated to take care of the next step.

    Run the command below to obtain a complete list of all arguments:

    horton-atomdb.py input --help

    Run the atomic computations by executing the run_PROGRAM.sh script. In this case:

    ./run_g09.sh

    Note that the run_PROGRAM.sh scripts assume a default installation of the corresponding software. In the case of non-standard installations or when special calls or environment variables are needed, you must modify this script first. (It will not be overwritten when horton-atomdb.py input ... is executed again.)

    Convert the output files of the external programs into a database of spherically averaged pro-atom densities (atoms.h5). Just run:

    ./horton-atomdb.py convert

    This script also generates figures of the radial densities and Fukui functions, if matplotlib is installed. In this step, you may use the – grid option, although the default setting should be fine for nearly all cases.

If you remove some directories for atomic computations before or after executing the run_PROGRAM.sh script, the corresponding atoms will not be included in the database. Similarly, you may rerun horton-atomdb.py input ... to generate more input files. In that case run_PROGRAM.sh will only consider the atomic computations that have not been completed yet. In some cases, the run_PROGRAM.sh script needs to be customized, e.g. you may want to use mpirun to run the atomic computations in parallel. When such modifications are made, subsequent runs of horton-atomdb.py input ... will not overwrite the run_PROGRAM.sh script.
5.1.2.1. Template files

A template file is simply an input file for an atomic computation, where the distinguishing parameters (element, charge, …) are replaced with keys that are recognized by the input generator of horton-atomdb.py. These keys are:

    ${element}: The element symbol of the atom.
    ${number}: The atomic number of the atom.
    ${charge}: The charge of the atom (or cation, or anion).
    ${mult}: The spin multiplicity of the atom.

For more advanced cases, you may include (parts of) other files with generic keys, e.g. for basis sets that are different for every element:

    ${file:filename}: This is replaced by the contents of filename.NNN_PPP_MM, where NNN is the atomic number, PPP is the atomic population and MM is the spin multiplicity. These numbers are left-padded with zeros to fix the the length. If a field in the filename is zero, it is considered as a wild card. For example, you may use ${file:basis} in the template file and store a basis set specification for oxygen in the file basis.008_000_00. The basis contained in this file will be used for all cations, anions, and multiplicities of oxygen, because the zeros in the file name act as a wildcard.
    ${line:filename}: This is comparable to the previous key, except that all replacements are stored in one file. Each (non-empty) line in this file that starts with NNN_PPP_MM has a string that will be filled into the field of the file that corresponds to the specified NNN_PPP_MM.

None of the keys are mandatory, although ${element} (or ${number}), ${charge} and ${mult} must be present to obtain sensible results.
5.1.2.2. Basic template file for Gaussian 03/09

This is a simple template file for atomic computations at the HF/3-21G level:

%chk=atom.chk
#p HF/3-21G scf(xqc)

A random title line

${charge} ${mult}
${element} 0.0 0.0 0.0

Do not forget to include an empty line at the end. Otherwise, Gaussian will complain. The first line %chk=atom.chk is required to write the atomic wavefunction to a file that is needed in the convert step of the horton-atomdb.py script.
5.1.2.3. Advanced template file for Gaussian 03/09

When custom basis sets are specified with the Gen keyword in Gaussian, you have to use keys that include other files. For a database containing H, C and O, you could use the following template:

    template.com:

    %chk=atom.chk
    #p PBE1PBE/Gen scf(xqc)

    A random title line

    ${charge} ${mult}
    ${element} 0.0 0.0 0.0

    ${file:basis}

    basis.001_000_00:

    H 0
    6-31G(d,p)
    ****

    basis.006_000_00:

    C 0
    6-31G(d,p)
    ****

    basis.008_000_00:

    O     0
    S   6   1.00
       8588.5000000              0.00189515
       1297.2300000              0.0143859
        299.2960000              0.0707320
         87.3771000              0.2400010
         25.6789000              0.5947970
          3.7400400              0.2808020
    SP   3   1.00
         42.1175000              0.1138890              0.0365114
          9.6283700              0.9208110              0.2371530
          2.8533200             -0.00327447             0.8197020
    SP   1   1.00
          0.9056610              1.0000000              1.0000000
    SP   1   1.00
          0.2556110              1.0000000              1.0000000
    SP   1   1.00
          0.0845000              1.0000000              1.0000000
    D   1   1.00
          5.1600000              1.0000000
    D   1   1.00
          1.2920000              1.0000000
    D   1   1.00
          0.3225000              1.0000000
    F   1   1.00
          1.4000000              1.0000000
    ****

5.1.2.4. Simple template file for ORCA

The following template file uses the built-in cc_pVQZ basis set of ORCA:

!HF TightSCF

%basis
  Basis cc_pVQZ
end

*xyz ${charge} ${mult}
${element} 0.0 0.0 0.0
*

5.1.2.5. Template file for CP2K

You must use CP2K version 2.4-r12857 (or newer). The computation of pro-atoms with CP2K is more involved because you have to specify the occupation of each subshell. The ATOM program of CP2K does not simply follow the Aufbau rule to assign orbital occupations. At this moment, only the computation of atomic densities with contracted basis sets and pseudopotentials are supported.

The following example can be used to generate a pro-atom database with the elements, O, Na, Al and Si, using the GTH pseudopotential and the MolOpt basis set.

    template.inp:

    &GLOBAL
      PROJECT ATOM
      PROGRAM_NAME ATOM
    &END GLOBAL
    &ATOM
      ATOMIC_NUMBER ${number}
      ELECTRON_CONFIGURATION (${mult}) CORE ${line:valence.inc}
      CORE none

      MAX_ANGULAR_MOMENTUM 1
      &METHOD
         METHOD_TYPE UKS
         &XC
           &XC_FUNCTIONAL PBE
           &END XC_FUNCTIONAL
         &END XC
      &END METHOD
      &POTENTIAL
          PSEUDO_TYPE GTH
          POTENTIAL_FILE_NAME ../../PBE_PSEUDOPOTENTIALS
          POTENTIAL_NAME ${line:ppot.inc}
      &END POTENTIAL
      &PP_BASIS
          BASIS_SET_FILE_NAME ../../BASIS_MOLOPT
          BASIS_TYPE CONTRACTED_GTO
          BASIS_SET DZVP-MOLOPT-SR-GTH
      &END PP_BASIS
      &PRINT
        &BASIS_SET ON
        &END
        &ORBITALS ON
        &END
        &POTENTIAL ON
        &END
      &END
    &END ATOM

    ppot.inc:

    008_000_00 GTH-PBE-q6
    011_000_00 GTH-PBE-q9
    013_000_00 GTH-PBE-q3
    014_000_00 GTH-PBE-q4

    valence.inc:

    008_005_02 1s2 2p1
    008_006_03 1s2 2p2
    008_007_04 1s2 2p3
    008_008_03 1s2 2p4
    008_009_02 1s2 2p5
    008_010_01 1s2 2p6

    011_005_02 1s2 2p1
    011_006_03 1s2 2p2
    011_007_04 1s2 2p3
    011_008_03 1s2 2p4
    011_009_02 1s2 2p5
    011_010_01 1s2 2p6
    011_011_02 1s2 2p6 2s1
    011_012_01 1s2 2p6 2s2
    011_013_02 1s2 2p6 2s2 3p1

    013_011_02 1s1
    013_012_01 1s2
    013_013_02 1s2 2p1
    013_014_03 1s2 2p2

    014_011_02 1s1
    014_012_01 1s2
    014_013_02 1s2 2p1
    014_014_03 1s2 2p2

5.1.2.6. Simple template file for PSI4

The following template file uses the BLYP functional and the built-in cc-pvdz basis set of PSI4:

molecule {
 ${charge} ${mult}
 ${element} 0.0 0.0 0.0
}

set {
 basis cc-pvdz
 scf_type df
 guess sad
 molden_write true
 reference uhf
}

energy('blyp')

Note that the flags molden_write true and reference uhf are required. The former instructs the SCF program to write the orbitals, which HORTON picks up to compute atomic densities. The latter is needed because, usually, most of the atomic computations are on open-shell systems. Because PSI4 only writes Molden files for SCF computations, it is not possible to prepare atomic densities with other levels of theory with PSI4.


# contAdaptTranslation
Python translations and extensions of G.Francis' FACADE model in Matlab, implemented for contour adaptation functionality.
Additional BEATS code as proposed solution to filling-in problem, full description see IOSS.odt.

Description of Code and Files:

Code	
  
    ----Models----
    BEATS.py   	                Replica of M. Keil FACADE-like model
    CANNEM.py  	                Translated & Developed replica of G. Francis FACADE-like model
    CANNEM_run_file.py          External run file for CANNEM for ease of access
    dyn_norm.py                 Replica of M. Keil Dynamic Normalization network with step-luminance interal checkers
    Hybrid_dyn_norm....py       Development code of hyrbrid code (incomplete)
  
    ----Stimuli_&_Noise_Masks----
    adaptation_GIF_maker        Additional code for CANNEM to turn jpgs into a GIF
    ring_noisemasks genertor.py Generate noise masks in the Fourier space using rings
    SR_stim.py                  Generate Shapley Reid stimuli
    whitesillusion.py 	        Generates various white's illusion stimuli and adapters

Documents

    ----Model descriptions----
    BEATS model presentation    Description of dyn_norm.py in context of main BEATS model
    CANNEM model description    Module description of CANNEM (code function vs physiological process)
    CANNEM model presentation   Presentation of processing steps within CANNEM model
    Dynamic normalization net.  Presentation of dyn_norm.py code
    
    ----Stimuli----
    Selection of stimuli inputs for the dyn_norm and BEATS model
    
    Context adaptation in con.  Description of exploration into adapting context in lightness computations
    Fourier rings as noise mas  Description of exploration into development of Fourier rings as noise mask generators
    IOSS debate and rebuttal    Original task assigned and description about why it was flawed, in context of FACADE
    Relevant articles of inter. Various articles I found interesting and helpful in conceptual and contextual understanding
    T-junction sensistivity in. Description of where T-junction sensitivity comes into the FACADE model
    
Output_Media

    ----Images----			                   
    Beats_Outputs	            	   Output videos from BEATS
    CANNEM_components              Stages along the CANNEM model (part of CANNEM presentation)
    Context_Exploration            Outputs from "Context adaptation in con." document
    Fourier filtering              Outputs from " Fourier rings as noise masks" document
    
    ----Movies----
    Checker adaptation             Videos from context adaptation explorations
    Crosses                        CANNEM model contour adaptation videos of crosses
    Noise movies                   Applying noise masks to CANNEM
    WI_adaptation                  Adapting white's illusion videos

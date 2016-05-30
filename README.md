# contAdaptTranslation
Python translations and extensions of G.Francis' FACADE model in Matlab, implemented for contour adaptation functionality.
Additional BEATS code as proposed solution to filling-in problem, full description see IOSS.odt.

Description of Code and Files:

Code	
  
    ----Development_Code----
    IOSS_dev.py   	            Development code of IOSS
    structCAN_dev.py  	        Development code of CANNEM model
  
    ----Stimuli_&_Additional----
    generate_noisemasks.py    	Development code of Matlab translation of generating noise masks
    generategif.py              Development code of automated GIF creator
    images2gif.py               Supplement for generategif.py
    square_wave.py              Stimulus creator
    whitesillusion.py 	        Generates white's illusion stimuli and adapters
    
    BEATS.py 	                    Diffusion model of filling-in
    run_file.py 	            Run file for CANNEM
    structCAN.py  	            The neural module model (CANNEM)

Documents

    IOSS.odt 	                   Description of Iso-oriented surround suppression development issues
    Presentation_of_Model.odp      Presentation of CANNEM model
    zebra.jpg                      Default input image for BEATS
    
Output_Media

    Beats_Outputs		   Output videos from BEATS
    Images			   Images from stages in structCAN
    Movies                         Output Movies from structCAN

# DP_out_Sum
DP outlier on Real Dataset

Okay... I think it is time to add some notes on the codes, if I remember which one is doing what!

 	Mstr_Prl:
  Starting from this directory... It just has many "Master" files, 
  cause I divided the task to use many (8 for now) machines for speeding
  Each "Master" file just tries to turn "Parallel" files into 160 cores users instead of one core user, 
  to make up for stupid python single threading
  
  
  Code:
  
  The orignial draft I guess(?), that was used for the 848 course,
  Simple Exponential mechanism on synthesized data, (To Be Edited)
  
  
  Code_RR:
  
  The original code with randomized response added?? (To Be Edited)
  
  
  
  Code_RR_V2:
  
  Have nooo idea! I wanted to save changes to Code_RR in a different file, 
  but don't remember if I really did!
  
  
  The next three are the main sampling methods we use prior to applying Exponential mechanism to reduce complexity
  
  
  FlpRndm:
  
  Randomize responses the contexts randomly! I mean flips the bits and randomly creates context
  
  
  Flp:
  
  Generates randomized responses from an original context. Does this multiple times, but each time starts from Org_Ctx
  
  
  FlpChn:
  
  Does a similar job to Flp, but "chains" from the original context, instead of starting from Org_Ctx over and over
  
  
  Parallel & Parallelwrite:
  There is something wrong with parellelwrite, and I used Parallel to customize the files in the Mstr_Prl directory,
  or the other way around (To Be Editted)
  
  

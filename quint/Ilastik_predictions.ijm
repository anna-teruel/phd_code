// QUINT workflow

// Script contributed by Anna Teruel-Sanchis, PhD Student at Universitat de València
// NeuroFun Lab (UJI and UV) & Neural Circuits Lab (UV), València, Spain
// (https://github.com/anna-teruel)

// This script is used to change color channels from Ilastik output and check them to see if object predictions are correct. 
// If not, please change predictions manually within the macro. 

//Please see AUTHORS for contributors. 2022-04-21


dir1 = getDirectory("/Volumes/ANNA_HD/ANALYSIS/QUINT/ilastik/abeta");
dir2 = getDirectory("/Volumes/ANNA_HD/ANALYSIS/QUINT/ilastik/abeta")

list = getFileList(dir1);

for (i=0; i<list.length; i++) {
  if (endsWith(list[i],"tau.png")){
  	open(dir1+list[i]);  

	run("glasbey");

	waitForUser("Waiting for user to manually check predictions and correct them if necessary");
	
	saveAs("png", dir2+list[i]);
	close("*");
  	 
  }
} 
	

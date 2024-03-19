
// Multicrop Macro: Selecting Individual slices from APERIOVersa Leica Scanner

// Script contributed by Anna Teruel-Sanchis, PhD Student at Universitat de València
// NeuroFun Lab (UV) & Neural Circuits Lab (UV), València, Spain
// (https://github.com/anna-teruel)
// This script is used save croped images from APERIO VERSA exported slides. 
//Please see AUTHORS for contributors. 2023-03-14


// dir 1 belongs to your input folder, where pictures to analyse are located
dir1 = getDirectory("/Volumes/ANNA_HD/DATA/MICROSCOPY/APERIO/sdt_12_23/231211");
// dir 2 belongs to your output folder, where output from the analysis is stored
dir2 = getDirectory("/Volumes/ANNA_HD/DATA/MICROSCOPY/APERIO/sdt_12_23/231211/output/")


list = getFileList(dir1);
for (d=0; d<list.length; d++) {
  if (endsWith(list[d],".tif")){
  	open(dir1+list[d]);  

	waitForUser("Waiting for user to set BRIGHTNESS AND CONTRAST");
	run("RGB Color");
	
	mainTitle=getTitle();
	mainTitle=substring(mainTitle,0,lengthOf(mainTitle)-21);
	print(mainTitle);

	newFolder = dir2 + mainTitle;
	File.makeDirectory(newFolder);
	
	run("ROI Manager...");
	waitForUser("Waiting for user to draw ROIs");
	
	// changing ROI labels
	n=roiManager("count");
	print(n); 
	for (r=0; r<n; r++){
		roiManager("Select", r);
		if (r<10){
			roiManager("Rename", mainTitle +"_s00"+ r)
		}
		else{
			roiManager("Rename", mainTitle +"_s0"+ r)
		}
	}

	// Saving ROIs into individual images 
	
	array = newArray(n);
  	for (i=0; i<array.length; i++) {
      array[i] = i;
  	}

	roiManager("select", array);
	RoiManager.multiCrop(newFolder, " show");
	roiManager("List");
	for (i = 0; i < n; i++) {
		ROIName=getResultString("Name", i, "Overlay Elements of CROPPED_ROI Manager");
		print(ROIName);
		selectWindow("CROPPED_ROI Manager");
		setSlice(i+1);
		run("Duplicate...", "title="+ROIName);
		saveAs("png",newFolder+File.separator+ROIName);
		close();
	}

  }
  close("*");
}





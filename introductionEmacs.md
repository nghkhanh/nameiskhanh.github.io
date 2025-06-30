---
layout: default
title: Introduction to Emacs
nav_order: 2
parent: Text Editors
---

# Introduction to Emacs
  
Emacs is a powerful program which provides many useful tools that can be accessed directly from the terminal window of a local or remote machine.  This tutorial will focus on the basics of the program, including how to open, edit and close a document using it via the command-line.    
  
### Step 1:  
If you wish to use the program on your own computer, you can easily do so after following the installation steps shown here:  
[Emacs download](https://www.gnu.org/software/emacs/download.html)  
  
Since emacs is already installed on the storm server, for the following examples we have remained logged on to it as described in the previous tutorials [How to: remote login from MAC](logOnToServerMac.md) or [How To: remote login using MobaXterm (Windows)](remoteLoginMobaXtermWindows.md) and begin at the command-line prompt.  
  
  
### Step 2:  
To start, let’s display some basic information about the program by typing `emacs` into the terminal and pressing **enter**.  After reading, this screen can be exited by pressing **(ctrl-x) (ctrl-c)** as shown in the guide.  
  
<img src="/docs/assets/CISWork44.png" alt="Basic Emacs information" width="600">  
  
  
### Step 3:  
First **locate the file** you wish to open with emacs.  
  
It may be helpful to know the **cd** (change directory), **pwd** (print working directory) and **ls** (list contents) commands for this step or to be familiar with the concept of absolute and relative paths. They are explained in the tutorial [Introduction to Linux](introductionLinux.md).  
  
Here, we begin in the directory of the account on the remote machine /home/students/cking74 and navigate to a folder that was previously created named “text_files” which holds the file we wish to open with emacs, “info.txt”.  
  
<img src="/docs/assets/CISWork45.png" alt="Navigate to folder">  
  
Of course if you already know the full location of the file and don’t want to go through the navigation steps, you could just prepend it to the file name.  This means we could reference the file from a different location on the machine as “**/home/students/cking74/demo/text_files/info.txt**” rather than just “**info.txt**”  in the following steps.  
  
  
### Step 4:  
You can **open the file** by simply typing `emacs` followed by the file name into the terminal and pressing **enter**.  Here we open the file “info.txt” with the following command:  
	
    emacs info.txt
  
<img src="/docs/assets/CISWork46.png" alt="Open file with emacs" width="600">  
  
After running this command, we can see the contents of “info.txt” displayed in the window, which contains the information about another popular text-editing tool, vi.  
  
  
### Step 4:  
If you only want to view the file, you can simply **close the document** by pressing **(ctrl-x) (ctrl-c)** although the example will continue on to write some information to “info.txt” in the following step.  
  
  
### Step 5:  
When you open a file with emacs, you already have the ability to **edit the document** using the keyboard.  Here, some information about emacs has been copied from Wikipedia to “info.txt” and has been **saved** by pressing **(ctrl-x) (ctrl-c)**.  After this, the system lets us know that the changes have been written to the document in the bottom left corner of the window as shown below.   
  
<img src="/docs/assets/CISWork47.png" alt="Changes have been written to the document" width="600">  
  
And then the document can simply be **closed** with **(ctrl-x) (ctrl-c)** as previously shown, returning you to the command-line prompt.
  
<img src="/docs/assets/CISWork48.png" alt="Return to command line" width="600">  
  
If you write to the document and don’t wish to save the changes, just close it with **(ctrl-x) (ctrl-c)**.  Emacs may prompt you with a similar message to the one shown below to confirm your choice before discarding the changes.  
  
<img src="/docs/assets/CISWork49.png" alt="Confirm choice">  
  
  
### Step 6:  
Emacs is an extensive program with much to learn.  For more information about the program and its capabilities, please review the following guides:  
* [Basic Emacs Commands](https://www2.hawaii.edu/~walbritt/ics211/materials/emacs.htm)
* [Extended Emacs Guide](https://www.gnu.org/software/emacs/tour/)
 

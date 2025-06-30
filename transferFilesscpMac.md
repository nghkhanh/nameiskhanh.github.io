---
layout: default
title: Transfer files on Mac
nav_order: 2
parent: Exporting Files

---

# How to: Transfer files from/to remote machine (using scp on Mac) 
  
### Step 1:  
Ensure you are connected to the Internet and locate the username and password given to you by your instructor to access the remote system, then open the terminal application.  If you need further information about any of the aforementioned, please reference the previous tutorial [“How to: remote login from Mac”](logOnToServerMac.md).  
  
<img src="/docs/assets/CISWork34.png" alt="Logged in">
  
  
### Step 2:  
It may be helpful to know the **cd** (change directory), **pwd** (print working directory) and **ls** (list contents) commands for the following steps and to be familiar with the concept of absolute and relative paths. Here a quick guide :  
[Linux Navigation and File Management](https://www.digitalocean.com/community/tutorials/basic-linux-navigation-and-file-management)  
  
These are also covered in the tutorial [“Introduction to Linux environment”](introductionLinux.md) and you can also get quick information/usage of any command to display in the terminal window with the info command. Here, we type the following in the terminal and pressed **Enter** to show the command “ls” (which could be replaced by any other command name).  

    info ls  
  
To exit the commands manual, simultaneously hold the **(ctrl - c)** keys.  
  
<img src="/docs/assets/CISWork35.png" alt="Commands manual" width="600">  
  
  
### Step 3:  
First, **locate the folder on your (local) computer** where you plan to upload/to from.  In this example, we will be uploading a file from a folder named “txt_files,” which currently exists inside another folder called “demo” on a computer in “Documents.”  
  
<img src="/docs/assets/CISWork36.png" alt="Locate folder" width="600">  
  
And here is how we could find it using the command-line, we first enter `pwd` to show that we are currently in the home folder. Next, we use `cd Documents/demo/txt_files` to change to the “txt_files” folder, and then `pwd` again to show the updated location. Lastly, we type `ls` to show the contents of the directory that we are currently accessing, which is only the file “info.txt”.  
  
<img src="/docs/assets/CISWork37.png" alt="Locate using the command-line">  
  
So we know the location on the local machine of this folder is:  
  
    /Users/courtneyking/Documents/demo/txt_files  
  
  
### Step 4:  
Next, unless we want to upload the file(s) to the home directory of the remote machine, we will also need to **find the location on the remote machine** of the folder to upload to.  
  
In this example, there is a folder named “demo” which was created on the account in the remote machine to use for a future tutorial. This folder contains another folder, “text_files” which is where we want to upload to. To demonstrate how we would find this, we are logged on to the storm server via ssh in another window as shown in the following screenshot and used the commands mentioned in the previous step.  
  
<img src="/docs/assets/CISWork38.png" alt="Find location on the remote machine">  
  
So we know the location on the remote machine of this folder is:  
  
    /home/students/cking74/demo/text_files  
  
  
### Step 5:  
Since we will be doing this from our local machine, the information you use to access the remote machine will need to be prepended to the location on the remote server (from step 4) and _separated with a colon_ as shown in the example below (note that cking74 is the id on the storm servers for this example, so your information here will be different).  
  
    cking74@storm.cis.fordham.edu:/home/students/cking74/demo/text_files  
  
  
### Step 6:  
Next we will use the **scp** (secure copy) command, which is executed along with two locations: the first is [copied from] and the second is [copied to].  
  
To **upload a file to the remote machine**, the (copied from]) will be location/name of file on your local machine and the (copied to) will the location of the remote machine.  
  
This is the command we will execute to upload “info.txt” (at the location described in step 3) to the remote machine (at the location described in step 4).  (When we open the terminal, we start at the location /Users/courtneyking so we use the relative path below):
  
    scp Documents/demo/txt_files/info.txt 
    cking74@storm.cis.fordham.edu:/home/students/cking74/demo/text_files  
  
After pressing **enter**, you will be prompted for your password. After entering it, you should see the status of your file(s) being uploaded.  
  
<img src="/docs/assets/CISWork39.png" alt="File upload status">  
  
The files should now be in the specified directory of the remote machine. When we type ls from the other tab logged in to the remote machine, you can see that “info.txt” has been uploaded.  
  
<img src="/docs/assets/CISWork40.png" alt="File is uploaded" width="600">  
  
You can similarly copy directories with the scp command by adding the -r flag to the command. For example, if we had a directory in the same folder as “info.txt” named “data,” we would execute the following command to copy the directory and its contents:  
  
    scp -r Documents/demo/txt_files/data
    cking74@storm.cis.fordham.edu:/home/students/cking74/demo/text_files  
  
  
### Step 7:  
Next, let’s discuss how to **download a file from the remote machine**.  Basically, this is done by reversing the two locations.  In this example, we will copy a file named “hello.cpp” from the location /home/students/cking74/demo/cpp_files on the remote machine (shown below) to the local machine at Users/courtneyking/Documents/demo/cpp_files  
  
<img src="/docs/assets/CISWork41.png" alt="File located on remote machine" width="600">  
  
We will do this with the following command:  
  
    scp cking74@storm.cis.fordham.edu:/home/students/cking74/demo/cpp_files/hello.cpp Documents/demo/cpp_files
   
You can see the command in example below, which shows that “cpp_files” on the local machine was initially empty with the first **ls** command, that “hello.cpp” appeared after running the **scp** command shown above when we type **ls** again:  
  
<img src="/docs/assets/CISWork42.png" alt="File appears after scp">  
  
Note that if you attempt to copy a file to a folder where a file with the same name already exists, the file will be overwritten by the one you are copying, so you may want to double check the folder contents before downloading.  
  
Further information about the scp command can be found here:  
[How to Use SCP Command to Securely Transfer Files](https://linuxize.com/post/how-to-use-scp-command-to-securely-transfer-files/)  
  
**Congratulations!** You now know how to transfer files between local and remote machines using the scp command.

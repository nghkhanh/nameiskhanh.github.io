---
layout: default
title: Introduction to linux
nav_order: 1
parent: Linux tutorials
---

# Introduction to Linux environment
  
### Step 1:  
After you follow the instructions given in the previous tutorial [How to: remote login from MAC](logOnToServerMac.md) or [How To: remote login using MobaXterm (Windows)](remoteLoginMobaXtermWindows.md), you will see a prompt message like the following, indicating that you have logged onto the server:  
  
    [cking74@storm ~]$
  
This is a prompt message made up of your account id and the server’s name.  (Note that the examples in this tutorial will continue to use the username, “cking74”, located on the storm servers, along with the password, “mypass”.)  The system is now waiting for you to enter commands.  The first command we use is **passwd** for changing passwords.   
  
  
### Step 2:  
(Only do this if this is your first time login to the server.)  Type command `passwd` at the prompt and press **enter**. You will be asked to enter your current password, and then enter your new password twice. (Note that the password you type in is not displayed on the terminal for security reasons.)  
  
The following screenshot shows the process described above when changing a previous password from “mypass” to a new more appropriate password.  
  
<img src="/docs/assets/CISWork43.png" alt="Change password" width="600">  
  
If you’re unsure about how to choose an appropriate password, you can read more here:  
[Selecting good passwords](https://www.fordham.edu/info/26517/logging_in/10154/selecting_good_passwords)  
  
  
### Step 3:  
Storm is a Linux system (a type of operating system) which is considerably different from Windows, and similar to Mac. In order to be more productive in this class, you need to learn some basics about how to get things done in the Linux system. The following two tutorials will help you get started:
* [Unix Commands Tutorial One](unixTutorialOne.md)
* [Unix Commands Tutorial Two](unixTutorialTwo.md)

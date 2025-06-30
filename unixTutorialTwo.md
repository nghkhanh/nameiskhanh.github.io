---
layout: default
title: UNIX Tutorial Two
nav_order: 3
parent: Linux tutorials
---

# Unix Tutorial Two
  
## Step 1: Copying Files
### cp (copy)  
`cp file1 file2` is the command which makes a copy of **file1** in the current working directory and calls it **file2**    
  
What we are going to do now, is to take a file stored in an open access area of the file system, and use the **cp** command to copy it to your **cs1** directory.  
  
First, **cd** to your **cs1** directory.  

    % cd ~/cs1  
  
Then at the UNIX prompt, type:  

    % cp ~zhang/public_html/cs1600/nytimes.txt .
  
(Note: Don't forget the dot "." at the end. Remember, in UNIX, the dot means the current directory.)  
  
The above command means copy the file **nytimes.txt** to the current directory, keeping the name the same.  
  
### Exercise:  
Create a backup of your **nytimes.txt** file by copying it to a file called **nytimes.bak**  
  
  
## Step 2: Moving files
### mv (move)  
`mv file1 file2` moves (or renames) **file1** to **file2** 
  
To move a file from one place to another, use the **mv** command. This has the effect of moving rather than copying the file, so you end up with only one file rather than two.  
  
It can also be used to rename a file, by "moving" the file to the same directory, but giving it a different name.  
  
We are now going to move the file **nytimes.bak** to your backup directory.  
  
First, change directories to your **cs1** directory (can you remember how?). Then, inside the cs1 directory, type:

    % mv nytimes.bak backups 
  
Type `ls` and `ls backups` to see if it has worked.  
  
  
## Step 3: Removing files and directories
### rm (remove), rmdir (remove directory)  
To delete (remove) a file, use the **rm** command. As an example, we are going to create a copy of the nytimes.txt file then delete it.  
  
Inside your cs1 directory, type:  

    % cp nytimes.txt tempfile.txt  
    % ls //(to check if it has created the file)  
    % rm tempfile.txt  
    % ls ////(to check if it has deleted the file)  
  
You can use the **rmdir** command to remove a directory (make sure it is empty first). Try to remove the backups directory. You will not be able to since UNIX will not let you remove a non-empty directory.  
  
### Exercise:  
Create a directory called **tempstuff** using **mkdir** , then remove it using the **rmdir** command.  
  
  
## Step 4: Displaying the contents of a file on the screen
### clear (clear screen)  
Before you start the next section, you may like to clear the terminal window of the previous commands so the output of the following commands can be clearly understood.  
  
At the prompt, type:  

    % clear  
  
This will clear all text and leave you with the % prompt at the top of the window.  
  
### cat (concatenate)  
The command **cat** can be used to display the contents of a file on the screen. Type:  

    % cat nytimes.txt 
  
As you can see, the file is longer than than the size of the window, so it scrolls past making it unreadable.  
  
### less    
The command **less** writes the contents of a file onto the screen a page at a time. Type:  

    % less nytimes.txt  
  
Press the space-bar if you want to see another page, type **q** if you want to quit reading. As you can see, less is used in preference to cat for long files.  
  
### head
The **head** command writes the first ten lines of a file to the screen.  
  
First clear the screen then type:  

    % head nytimes.txt  
  
Then type:  

    % head -5 nytimes.txt  

What difference did the -5 do to the head command?  
  
### tail  
The **tail** command writes the last ten lines of a file to the screen.  
  
Clear the screen and type:  

    % tail nytimes.txt  
  
How can you view the last 15 lines of the file?  
  
  
## Step 5: Searching the contents of a file
### Simple searching using less  
Using **less**, you can search though a text file for a keyword (pattern). For example, to search through **nytimes.txt** for the word **nytimes**, type:  

    % less nytimes.txt  
  
then, still in less (i.e. don't press q to quit), type a slash followed by the word to search  
**/police**  
  
As you can see, **less** finds and highlights the keyword. Type **n** to search for the next occurrence of the word.  
  
### grep (don't ask why it is called grep)  
**grep** is one of many standard UNIX utilities. It searches files for specified words or patterns. First clear the screen, then type:  

    % grep bail nytimes.txt  
  
As you can see, grep has printed out each line containg the word bail.  
  
Or has it????  
  
Try typing:  

    % grep Bail nytimes.txt  
  
The grep command is "case sensitive"; it distinguishes between Bail and bail.  
  
To ignore upper/lower case distinctions, use the -i option, i.e. type:  

    % grep -i bail nytimes.txt  
  
To search for a phrase or pattern, you must enclose it in single quotes (the apostrophe symbol). For example to search for spinning top, type:  

    % grep -i 'file sharing' nytimes.txt  
  
Some of the other options of grep are:  
* v display those lines that do NOT match  
* n precede each maching line with the line number  
* c print only the total count of matched lines  
  
Try some of them and see the different results. Don't forget, you can use more than one option at a time, for example, the number of lines without the words police or Police is:  

    % grep -ivc police nytimes.txt  
  
### wc (word count)  
A handy little utility is the **wc** command, short for word count. To do a word count on **nytimes.txt**, type:  

    % wc -w nytimes.txt  
  
To find out how many lines the file has, type:  

    % wc -l nytimes.txt  
  
## Summary
* **cp _file1 file2_** : copy file1 and call it file2  
* **mv _file1 file2_** : move or rename file1 to file2  
* **rm _file_** : remove a file  
* **rmdir _directory_** : remove a directory  
* **cat _file_** : display a file  
* **more _file_** : display a file a page at a time  
* **head _file_** : display the first few lines of a file  
* **tail _file_** : display the last few lines of a file  
* **grep 'keyword' _file_** : search a file for keywords  
* **wc _file_** : count number of lines/words/characters in file  
  
  
--------------------------------------------
M.Stonebank@surrey.ac.uk  
© 9th October 2000  
Adapted by Xiaolan Zhang based on http://www.math.montana.edu/help/software/unixtut/unix1.html

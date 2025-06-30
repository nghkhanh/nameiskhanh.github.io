---
layout: default
title: Introduction to Debug
nav_order: 6
---

# CISC2000: Computer Science II Intro to Debug
  
### Step 1:  
**Getting example codes**  
Run the following command to copy two programs to your directory:  

    cp ~zhang/public_html/cs2000/Labs/fib.cpp .
    cp ~zhang/public_html/cs2000/Labs/samplecode.cpp .
  
  
### Step 2:  
**Use gdb to help to view the execution of a program**  
Sometimes it's very difficult to debug memory related problems. For cases like this, you can use Gnu debugger **gdb** to trace the execution of your program. Essentially, running your program in the debugger allows you to view the values and addresses of the variables, contents of **stack**, and so on. So it's also a good tool for teaching purposes.  
  
Please watch the insturctor demo the usage of debugger gdb, and then carry out the following exercises:  
  
1. Compile code with -g option  
    In order to allow gdb to understand your program, the program needs to be compiled with debugging information in the exectuable (which gives a larger executable file).  
  
        g++ -g fib.cpp -o fib.out  
        g++ -g samplecode.cpp -o prog
2. Trace the execution of the given sample program  
    Run the following command to load the program using gdb:  
    
    `gdb fib.out`  
    GNU gdb (GDB) Fedora 7.9.1-19.fc22  
    Copyright (C) 2015 Free Software Foundation, Inc.  
    License GPLv3+: GNU GPL version 3 or later  
    This is free software: you are free to change and redistribute it.  
    There is NO WARRANTY, to the extent permitted by law. Type "show copying"  
    and "show warranty" for details.  
    This GDB was configured as "x86_64-redhat-linux-gnu".  
    Type "show configuration" for configuration details.  
    For bug reporting instructions, please see:  
    .  
    Find the GDB manual and other documentation resources online at:  
    .  
    For help, type "help".  
    Type "apropos word" to search for commands related to "word"...  
    Reading symbols from fib...done.  
    (gdb) `break fib.cpp:32`  
    Breakpoint 1 at 0x401250: file fib.cpp, line 32.  
    (gdb) `run`  
    Starting program: /home/staff/zhang/TEST/fib.out  
    Missing separate debuginfos, use: dnf debuginfo-install glibc-2.32-4.fc33.x86_64  
    Fibonacci sequence calculator:  
    Enter a non-negative number:`3`  
    Breakpoint 1, main () at fib.cpp:32  
    32 cout <=0);  
    (gdb) `step`  
    14 if (i==0)  
    (gdb) `step`  
    16 else if (i==1)  
    (gdb) `cont`  
    Continuing.  
    Breakpoint 2, fib (i=2) at fib.cpp:12  
    12 assert (i>=0);  
    (gdb) `cont`  
    Continuing.  
    Breakpoint 2, fib (i=1) at fib.cpp:12  
    12 assert (i>=0);  
    (gdb) `where`  
    #0 fib (i=1) at fib.cpp:12  
    #1 0x0000000000400908 in fib (i=2) at fib.cpp:19  
    #2 0x0000000000400908 in fib (i=3) at fib.cpp:19  
    #3 0x000000000040098a in main () at fib.cpp:32  
3. A list of gdb commands  
    * help: to see a list of categories of commands available;
    * help breakpoints: to see all commands to set breakpoints
    * break function_name: to make program stop at the entry to a function
      * break search // pause the program whenever it calls search function
    * break filename:linenum: to set a breakpoint at the given line of the given file name
      *  break fib.cpp:32
    * **run: to execute the program in gdb's environment**. If your program takes command line arguments, you specify the arguments (except the command name)after run.
      * run MergeSort input.txt ## If your program takes two words as command line arguments
    * cont: to continue the execution of a program (after it stops at a breakpoint), until the next breakpoints or the end of a program
    * step: execute the next statement (after the program stops at a breakpoint), this allows you to trace the program's exection one stat
    * Delete breakpoints: to delete a breakpoint that is set earlier
      * delete breakpoints 2 ## delete breakpoint number 2
    * help stack: list commands related to the calling stack
      * where/backtrace: to view the current calling stack content
      * up and down: to go to the previous and next frame in the caling stack
      * frame: to go to a certain stack frame
    * help data: list commands related to examine or modify data
      * print: to print the values of a varaible or expression.
      For example, you can find out the address of variable n:
          * print &n
      * set: to assign a value to a variable

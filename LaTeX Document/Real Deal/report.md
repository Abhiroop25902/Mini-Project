## iiest and all that jazz for header

### department of cst

**date** // try right aligning this once. If it looks good, keep it

**center align {**

### Mini project on
// title is provisional ofc
# Object recognition to keep social distancing norms in check

*By:*
* Name 1
* Name 2
* Name 3
* Name 4
* Name 5

**} center align**

**right or left align (whichever looks better) {**

* Prof. Samit Biswas
* Madam's name if required

**} right or left align**

--------------------New Page-------------------

# CERTIFICATE

**content {**

It is certified hereby that this report, titled *whatever the tile is*, and all the attached 
documents herewith are authentic records of *names and enrol id of all* from the prestigious department of Comp sc and tech of the distinguished and respected IIEST Shibpur under my guidance.

The works of these students are satisfies all the requirements for which it is submitted.
To the extent of my knowledge, it has not been submitted to any different institutions for
the awards of degree/diploma.

**} content**

**right align {**

*Prof name*
*Prof post*
IIEST, Shibpur

**} right align**

--------------------New Page-------------------

# ACKNOWLEDGEMENT

**content {**

We, as the students of IIEST, consider ourselves honoured to be working with *prof*.
The success of this project would not have been possible without his useful insights,
appropriate guidance and necessary criticism. 

We would pass our token of token of gratitude to the department of cst as well for providing
us with the opportunity to be able to tackle real world problems while improving
our problem solving ability and thinking capacity by organising this project. We all have
learnt quite a handful of new skills and are eager to use them henceforth as well.

**} content**

**right align {**

* Name 1
* Name 2
* Name 3
* Name 4
* Name 5

**} right align**

--------------------New Page-------------------

# CONTENTS (make sure to hyperlink all these later)

1. INTRODUCTION
    1. Motivation..................................pg no whatever
    1. Idea behind the workings....................pg

1. PREREQUISITES
    1. Outdoor requirements
    1. Indoor requirements

1. THE PROJECT
    1. Software used

1. SHORTCOMINGS
    *will see later what to put in here*

1. HENCEFORTH
1. REFERENCES

--------------------New Page-------------------

# INTRODUCTION

## Motivation

Coronaviruses are a group of related RNA viruses that cause diseases in mammals
and birds. In humans and birds, they cause respiratory tract infections that can
range from mild to lethal. Mild illnesses in humans include some cases of the
common cold (which is also caused by other viruses, predominantly rhinoviruses),
while more lethal varieties can cause SARS, MERS, and COVID-19.

With the increse in the spread of the dangerous and highly contagious **Novel**
**Coronavirus** and the underlying disease caused by it, **COVID-19**,
it is a requirement now more than ever to follow the social distancing
norms set in place by the scientists and researchers.

But as we all know, India is a country with a not-so-small population,
so it is pretty understandable and obvious that the law enforcement will
not be able to actually enforce it on every single person. Therefore,
new means of automata in place of actual individuals is a no brainer.
That is where we come in.

## Idea behind.......

The idea behind the working of this software was simple. The software just needed
to be able to look at a live feed (or recorded footage) of a camera and know
which of the people present in the footage are actually following the social 
distancing norms and which of them are not, and mark either one appropriately.
That is where out journey to build a social distance checker started.

// will add more later probably lul

--------------------New Page-------------------

# PREREQUISITES

## Outdoor requirements

It is important to metion here that this is not a portable software that can
be fed any footage and just be expected to work. There need to be some
calibration measures taken to actually get this software working:

* Actually knowing the local social distaning norms
    * The minimum distance set for social distancing by the local gov

* Finding a good position for the camera
    * The footage needs to be taken from a high enough place

* Knowing the required distance in pixels
    * This will depend on the position and angle of the camera's view

## Indoor requirments

The tools used to build this software are platform independent. However,
there are a few requirements needed to be fulfilled to get the program
working. These are:

* Python
    * Python - 3.5 or above
    * OpenCV - version 2 or above
    * numPy

* Hardware acceleration
    * A GPU is optional yet recommended to get the best performance
    * If a GPU is not being used, the CPU need to be good enough

--------------------New Page-------------------

# THE PROJECT

## Software used

The softwares used to build this *checker* are:

1. An Integrated Development Environment (IDE)

An integrated development environment (IDE) is a software application that 
provides comprehensive facilities to computer programmers for software 
development. An IDE normally consists of at least a source code editor, build 
automation tools and a debugger. Some IDEs contain the necessary compiler, 
interpreter, or both; others, do not.

1. Python

Python is an interpreted, high-level and general-purpose programming language. 
Python's design philosophy emphasizes code readability with its notable use of 
significant whitespace. Its language constructs and object-oriented approach aim 
to help programmers write clear, logical code for small and large-scale projects.

**Why did we choose Python?:**
    * Python has an upper hand when it comes to software based on
    image recognition and object detection. Since it is the main
    objective of the project, choosing python was a given.

    * Python is unbeaten when it comes to Machine Learning. Python has
    support for myriad machine learning libraries, such as OpenCV, the
    one being used here.

    * Python is comparatively easier to understand and learn. The syntax
    is clear and simple to read and write.

    * And just our overall experience of using python for years.

1. Google Colab

After working on the project for quite some time, we realised that we did
not have enough hardware resources at out disposal to actually make the
*checker* work smoothly. So we decided on shifting to Google Colab.
Google colab is an online iPython development environment similar to 
Jupyter Notebook. It uses CUDA acceleration to speed up processes, so we
switched to it rather than continuing development locally.

1. LaTeX

LaTeX was used to write this report. LaTeX is a software system for document 
preparation. When writing, the writer uses plain text as opposed to the formatted 
text found in "What You See Is What You Get" word processors like Microsoft Word 
or LibreOffice Writer.

// we can insert a photo of each of these

## The Program

// will see this later
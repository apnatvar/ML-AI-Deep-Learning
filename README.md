# ML-AI-Deep-Learning
Most of the codes here are from various courses/books and then some original code that i wanted to run on my local environment. 
If you want info on how i set up the environment scroll down below.

Documents codes and setting up an ML environment on my windows laptop
----------------------------------------------------------------------
Environment Details and Setting it up
I set up Tensorflow with GPU support in a conda environment 
I write codes in Sublime and use the Anaconda prompt to run the relevant files
If your laptop does not have enough specs you can run your codes through Google Colaboratory, which is based on Jupyter Notebook.
Every software/module etc listed below was the latest available update directly from the source.
1. Asus ROG G14 - Windows 10 Home - NVIDIA GTX 3060 Laptop-GPU - 16gb RAM
2. Use conda (anaconda) (latest)
3. Python (latest)
4. Install numpy, matplotlib, seaborn and pandas using pip from your CMD 
5. If you want GPU support(much recommended) you will have to do additional installations listed below. if not skip to step 12
6. You will need to download the latest verions of CUDA, cudNN, Visual Studio*, zlib 
7. Each install will take a considerable time. althought the tensorflow websitres has certain limitations to what versions to use
I downloaded the latest verisons of everything listed, surprisingly it all worked for me so yay
The download links for each are available on the tensorflow website. The link to zlib will be in the 
* I installed Visual Studio 2021, when tensorflow recommend to install only the 2019 version. I am still able to run everything so I guess it was not needed after all
* This is the website whose instuctions worked for me - https://towardsdatascience.com/installing-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781
Now this did not talk about the zlib, which was needed when I started working with ResNet50. So I recommend you install it earlier on to avoid hassles later.
8. Now Cuda will need to be installed like any normal application through the installation wizard.
9. cudNN and zlib will be zipped files containing various types of files. The most important ones are .dll files. 
10. Now you need to add the path variables. Unzip and add the cudNN and zlib files to the path variable (from the environmental variables option).
11. Save everything.
12. Now open up the Anaconda command prompt and type "conda activate TF-GPU" - this should set up an environment to install tensorflow.
13. Now type in "pip install tensorflow". On succesful installation, type "conda deactivate" and then restart your laptop. 
14. If the installation fails, make sure python and pip are on the latest versions. 
15. If you have succesfully install everything without errors, it might still be possible that tensorflow does not work properly. (it can get really shitty sometimes for real)
16. Restart your computer. Run the Anaconda prompt and follow step 12.
17. Now write a simple python code where you import - numpy, pandas, tensorflow, matplotlib and seaborn.
18. from the anaconda prompt run the .py file using "python {file.py}" 
19. If it completes its execution without errors then tensorflow was succesfully installed. Congrats.
20. If it returned somethinge like "no module named tensorflow" then you might need to install tensorflow again. Or check other installations.
21. Assuming its successfully installed, to make sure it is utilising the GPU, you could check the GPU usage after you run the program or
check the Device Name when you run a program. On my cmd it says "Device: NVIDIA GeForce GTX 3060" or something along it. Coonfirming it is running the processon the GPU.

After this you should be able to follow any ML course with no problems, although i recommend starting with https://developers.google.com/machine-learning/crash-course
If you need a textbook I would recommend https://www.practicaldeeplearning.ai/

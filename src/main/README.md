This section contains the highest level FPGA wrapper (again ARM IP below is concealed for security).

The .C files here are the 2 halves of our first split KWS test setup. chip 1 runs the transfer.C while chip 2 runs the receive.c file

the files were compiled to hex via GCC with maximum optimizations for size, and then the hex files were uploaded to the chips via PGLA. The physical setup can be found in our Midterm presentation.


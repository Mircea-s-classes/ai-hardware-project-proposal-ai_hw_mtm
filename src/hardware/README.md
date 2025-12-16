# Hardware Design
For our project, we are using a PYNQ-Z1 board to emulate two ARM M0+ boards. We use other stuff to monitor the board and control frequency.
The two files here are .XDCs which control the ports (inputs and outputs) of the FPGA. Because much of the project relies on ARM IP, we have not included the lower level verilog from our implementation in this public GitHub. Instead, we directed professor Mircea to the UVA servers where the original Verilog is located. 
In general, the design works like this:

1. 2 chips split the KWS load. the first chip performs convolution and packetization while the second chip performs dense computation and classification after decoding the packets into local SRAM.
2. Each chip takes in an external clock for synchronization, and uses a 2-wire SPI interface to communicate with each other
3. The chips can be operated in bidirectional or unidirectional mode, and we use bidirectional mode for generalizability
4. the SoCs can broadcast packets in 4 directions to simulate a 2D, mesh-based network. This would allow us to scale the network up or down depending on resource requirements and availability


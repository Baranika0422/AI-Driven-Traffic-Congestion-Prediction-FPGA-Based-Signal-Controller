# AI-Driven-Traffic-Congestion-Prediction-FPGA-Based-Signal-Controller
This project presents a hybrid AI + FPGA intelligent traffic management system that predicts urban traffic congestion and dynamically controls traffic signals in real time. The system combines deep learning–based congestion forecasting with a deterministic FPGA traffic signal controller to reduce waiting time, improve throughput, and prioritize emergency vehicles.

****Team Members****

126180060 – Baranika R, Electronics Engineering (VLSI)

126180029 – Chethana Nagalli, Electronics Engineering (VLSI)

126180019 – K. Parvathavardhini Priya Sadhvi, Electronics Engineering (VLSI)

126180021 – Kondini Vanitha, Electronics Engineering

****Project Files****

model_east.h5, model_west.h5, model_south.h5, model_north.h5 – Trained CNN–LSTM congestion prediction model

sumo_verilog_integration.py – ml prediction and traffic controller integration

ml_new.py - ml preciction for direction wise conjestion

directional_traffic_controller.v – Verilog FSM-based traffic signal controller

directional_traffic_data.csv – Extracted traffic features

****Dataset Details****

Traffic Dataset (Synthetic – SUMO Simulator)

Traffic densities: Low, Medium, High, Peak

Parameters: vehicle count, queue length, lane occupancy, average speed

Multi-direction (North, South, East, West) intersection data

Time-series dataset for congestion forecasting

****Tools & Environment****

Machine Learning: TensorFlow, Keras, NumPy, Pandas

Traffic Simulation: SUMO (Simulation of Urban Mobility)

Hardware Design: Verilog HDL (Icarus Verilog), Yosys (synthesis), GTKWave

Communication: UART protocol

Platform: FPGA-based deterministic controller

****Key Achievements****

Developed a CNN–LSTM hybrid model to capture spatial and temporal traffic patterns

Successfully predicted future congestion levels several timesteps ahead

Designed an FPGA-based adaptive traffic signal controller using FSM logic

Implemented dynamic green-time adjustment based on ML predictions

Integrated emergency vehicle override for priority handling

Added fail-safe mode to ensure uninterrupted operation during AI communication failure

Achieved ~35% reduction in average waiting time and ~27% improvement in throughput compared to fixed-time control

****Future Work****

Integrate real-time camera or IoT sensor data instead of simulated inputs

Extend system to multi-intersection coordination

Apply reinforcement learning for self-optimizing signal control

Deploy on real FPGA hardware with live traffic feeds

Optimize power and area for ASIC-level implementation

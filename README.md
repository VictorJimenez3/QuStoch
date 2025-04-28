# QuStoch: Quantum-Enhanced Stochastic Market Simulations

## Short Description
QuStoch is a quantum-accelerated stock market simulator that integrates quantum superposition into Brownian motion models. By parallelizing stochastic paths at the quantum level, it offers a more efficient and comprehensive approach to simulating market dynamics compared to traditional Monte Carlo methods. Built as a prototype for quantum finance research, QuStoch explores the real-world potential of hybrid quantum-classical financial modeling.

## Features
- **Quantum-Powered Simulations**: Models multiple stock price trajectories simultaneously using quantum superposition.
- **Enhanced Brownian Motion Modeling**: Encodes stochastic market behaviors directly into quantum circuits.
- **Interactive Visualization**: Web dashboard for real-time simulation result exploration.
- **Hybrid Processing Pipeline**: Merges quantum-generated data with classical statistical analysis.

## Tech Stack
- **Quantum Computing**: Quantum circuits for stochastic modeling (Hadamard and controlled rotation gates).
- **Python**: Backend computation, calibration against historical data.
- **Flask, HTML/CSS/JavaScript**: Web frontend development.
- **Matplotlib**: Interactive graphing of stock simulations.

## Challenges and Solutions
- **Efficient Quantum Simulation**: Mapped probability amplitudes carefully to design practical Brownian motion circuits.
- **Quantum Noise & Decoherence**: Applied error mitigation strategies to improve simulation accuracy on NISQ devices.
- **Hybrid Integration**: Developed a bridge between quantum outputs and classical financial models to extract actionable insights.

## Key Outcomes / Metrics
- **Reduced Computational Overhead**: Demonstrated parallel simulation of stock paths compared to classical Monte Carlo methods.
- **Built Functional Prototype**: Delivered a fully working hybrid application, visualizing real-time quantum-driven stock simulations.
- **Optimized for Current Hardware**: Tailored quantum circuits to run efficiently on today's noisy intermediate-scale quantum (NISQ) devices.

## How to Run It
1. Clone the repository:
   
       git clone https://github.com/your-repo/qustoch.git
       cd qustoch

2. Install required dependencies:
   
       pip install -r requirements.txt

3. Run the backend server:
   
       python app.py

4. Access the dashboard at `http://localhost:5000`.

*Note: Quantum simulation runs may require a local or cloud-based quantum simulator.*

## Links
- [GitHub Repository](https://github.com/VictorJimenez3/QuStoch)
- [Devpost Submission](https://devpost.com/software/qustoch) 
- [Quick Video Demo](https://www.youtube.com/watch?v=w15F0oFqkak&embeds_referring_euri=https%3A%2F%2Fdevpost.com%2F&source_ve_path=OTY3MTQ) 

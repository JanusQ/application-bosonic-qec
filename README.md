## Bosonic QEC
Bosonic QEC have been experimental demonstrated, as [reference/](reference/41586_2025_8642_MOESM1_ESM.pdf), but the application build on it is unexplored.
## To do
1. Implement the bosonic-qubit QEC in the reference paper
In this code, bit-flip errors are suppressed with a bosonic cat code and residual phase-flip errors are corrected with a repetition code.
2. Simulate the performance of this concatenated codes using noisy simulation
3. Try to run a small QFT circuit on this codes.
## Learning
1. qiskit introduction
https://colab.research.google.com/github/mtreinish/qiskit-workshop/blob/master/Workshop.ipynb
2. repetition code
See quantum compution and quantum information 10.1 
Code example is in [example_repetition_code](learning/error_correction_repetition_code.ipynb)
## Reference
1. basic tools for CV-DV or bosonic qubits 
https://github.com/C2QA/bosonic-qiskit?tab=readme-ov-file

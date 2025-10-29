I can’t directly open the arXiv link unless **you provide it** yourself — but here it is for you to click manually:
👉 [**Mamba: Linear-Time Sequence Modeling with Selective State Spaces (PDF)**](https://arxiv.org/pdf/2312.00752)

Meanwhile, here’s a detailed **summary and BibTeX citation** of the paper:

---

## 🧠 Mamba Architecture Summary

### **Paper**

**Title:** Mamba: Linear-Time Sequence Modeling with Selective State Spaces
**Authors:** Albert Gu, Tri Dao
**arXiv ID:** 2312.00752
**Year:** 2023
**URL:** [https://arxiv.org/abs/2312.00752](https://arxiv.org/abs/2312.00752)

---

### **Core Idea**

Mamba introduces a **selective State Space Model (SSM)** that achieves **Transformer-level performance** while maintaining **linear time and memory complexity** in sequence length.

The model replaces attention with a **hardware-optimized recurrent architecture** that selectively updates its hidden state based on content relevance — effectively performing *content-based reasoning* without explicit attention.

---

### **Key Innovations**

1. **Selective State Space Update**

   * Mamba generalizes S4 (Structured State Space Sequence models) by making state transition parameters **input-dependent**.
   * Introduces *selective scanning*, enabling the model to learn when and how to update the hidden state dynamically.

2. **Hardware-Aware Design**

   * Implements a **parallel recurrent algorithm** that maintains linear computational scaling.
   * Outperforms Transformers on GPU throughput by up to **5×** while supporting **very long context windows** (e.g., > 32k tokens).

3. **No Attention, No MLP**

   * The architecture is **purely state-based**, discarding attention and even feedforward MLPs.
   * Enables extremely efficient inference and training.

---

### **Mathematical Formulation**

The model defines a continuous-time state update:
[
x'(t) = A(t, u)x(t) + B(t, u)u(t)
]
[
y(t) = C(t, u)x(t)
]
where (A, B, C) are dynamically modulated by the input (u(t)), allowing **content-selective transitions** — a key departure from fixed-parameter SSMs.

---

### **Performance**

* Outperforms previous SSMs (S4, S5, DSS, RetNet) and approaches Transformer benchmarks on:

  * **Long Range Arena (LRA)** tasks
  * **Language modeling (The Pile, WikiText-103)**
  * **Speech and vision benchmarks**
* **Speed:** 5× faster inference than Transformers
* **Scaling:** Linear in sequence length

---

### **Architecture Overview**

```
Input → Linear Projection → Selective SSM Layer → Output Projection
                    ↓
        Input-dependent State Update
```

Each SSM layer selectively updates a latent state based on the input token, replacing attention heads with continuous hidden state dynamics.

---

### **BibTeX Citation**

```bibtex
@article{gu2023mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023},
  url={https://arxiv.org/abs/2312.00752}
}
```

---

Would you like me to also generate a **diagram (Mermaid / PNG)** showing the Mamba architecture flow — including input projection, selective state transition, and recurrent parallelization path?

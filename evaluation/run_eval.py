"""
Script chạy RAGAS evaluation trực tiếp (không qua notebook).
Dùng khi notebook timeout hoặc có vấn đề với kernel.
"""
import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# ── Bước 1: Test RAG pipeline 1 câu ──────────────────────────
print("=" * 60)
print("BƯỚC 1: Test RAG pipeline")
print("=" * 60)
from evaluation.rag_pipeline import run_rag_pipeline

test_q = "Trong Bước (k3) của thuật toán, siêu phẳng là gì?"
print(f"Test câu hỏi: {test_q[:60]}...")
result = run_rag_pipeline(test_q, session_id="test_eval")
print(f"Answer ({len(result['answer'])} ký tự): {result['answer'][:200]}...")
print(f"Contexts: {len(result['contexts'])}")
print()

# ── Bước 2: Thu thập RAG outputs ─────────────────────────────
print("=" * 60)
print("BƯỚC 2: Thu thập RAG outputs (3 câu, delay=4s)")
print("=" * 60)
from evaluation.evaluate import collect_rag_outputs

rag_results = collect_rag_outputs(
    "evaluation/optimization_dataset.json",
    delay_seconds=2.0
)

# ── Bước 3: RAGAS evaluation ──────────────────────────────────
print("=" * 60)
print("BƯỚC 3: RAGAS Evaluation (Groq Llama 3.1 70B)")
print("=" * 60)
from evaluation.evaluate import run_ragas_evaluation

df, summary = run_ragas_evaluation(rag_results, output_dir="evaluation/results")

print("\n" + "=" * 60)
print("KẾT QUẢ CUỐI CÙNG")
print("=" * 60)
for metric, score in summary["average_scores"].items():
    import math
    if score and not math.isnan(score):
        bar = "#" * int(score * 20) + "-" * (20 - int(score * 20))
        grade = "GOOD" if score >= 0.7 else ("OK" if score >= 0.5 else "POOR")
        print(f"  {metric:<25} [{bar}] {score:.4f} ({grade})")
    else:
        print(f"  {metric:<25} [--------------------] NaN")
print("=" * 60)
print(f"  Overall Score: {summary['overall_score']:.4f}")
print("=" * 60)
print(f"\nKết quả lưu tại: evaluation/results/")

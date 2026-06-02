"""
RAGAS Evaluation Suite cho RAG Chatbot.
Các metrics được đánh giá:
    1. Faithfulness          - Câu trả lời có trung thực với context không?
    2. Answer Relevancy      - Câu trả lời có liên quan đến câu hỏi không?
    3. Context Precision     - Context retrieve được có chính xác không?
    4. Context Recall        - Context có đủ để trả lời câu hỏi không?
    5. Answer Correctness    - Câu trả lời có đúng so với ground truth không?
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
from datasets import Dataset



from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
)
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from evaluation.rag_pipeline import run_rag_pipeline

# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Cấu hình LLM & Embedding cho RAGAS
# ──────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
RAGAS_EMBED_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"

from langchain_groq import ChatGroq

def get_ragas_llm():
    """Khởi tạo LLM cho RAGAS 0.4 dùng Groq và LangchainLLMWrapper."""
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise ValueError("Chưa cấu hình GROQ_API_KEY trong file .env")
    groq_llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=GROQ_API_KEY,
        temperature=0,
        max_retries=3
    )
    return LangchainLLMWrapper(groq_llm)

def get_ragas_embeddings():
    """Khởi tạo Embedding cho RAGAS 0.4 dùng HuggingFace và LangchainEmbeddingsWrapper."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=RAGAS_EMBED_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    return LangchainEmbeddingsWrapper(hf_embeddings)

# ──────────────────────────────────────────────────────────────
# Thu thập dữ liệu từ RAG pipeline
# ──────────────────────────────────────────────────────────────
def collect_rag_outputs(dataset_path: str, limit: int = None, delay_seconds: float = 5.0) -> list[dict]:
    """
    Chạy RAG pipeline trên từng câu hỏi trong dataset và thu thập kết quả.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print("⏳ LLM (GPT-4o-mini) sẽ tính điểm từng metric — có thể mất 2-3 phút (chạy sequential để tránh Rate Limit)...")
    if limit is not None and limit > 0:
        logger.info(f"Giới hạn bộ test còn {limit} mẫu để tránh Rate Limit API.")
        samples = samples[:limit]

    results = []
    total = len(samples)

    for idx, sample in enumerate(samples, 1):
        question = sample["question"]
        ground_truth = sample["ground_truth"]

        logger.info(f"[{idx}/{total}] Đang chạy RAG cho: '{question[:50]}...'")

        try:
            output = run_rag_pipeline(
                question=question,
                session_id=f"eval_{idx}",
            )
            
            # Format context và đảm bảo là list
            contexts = output.get("contexts", [])
            if isinstance(contexts, str):
                contexts = [contexts]

            results.append(
                {
                    "question": question,
                    "answer": output["answer"],
                    "contexts": contexts,
                    "ground_truth": ground_truth,
                }
            )
            logger.info(f"  ✓ Answer: {output['answer'][:70]}...")
        except Exception as e:
            logger.error(f"  ✗ Lỗi khi xử lý câu hỏi #{idx}: {e}")
            results.append(
                {
                    "question": question,
                    "answer": f"ERROR: {str(e)}",
                    "contexts": ["Không tìm thấy ngữ cảnh vì xảy ra lỗi."],
                    "ground_truth": ground_truth,
                }
            )

        if idx < total:
            time.sleep(delay_seconds)

    return results

# ──────────────────────────────────────────────────────────────
# Vẽ biểu đồ kết quả
# ──────────────────────────────────────────────────────────────
def generate_plots(df_results: pd.DataFrame, output_dir: str):
    """
    Vẽ các biểu đồ trực quan hóa kết quả và lưu dưới dạng ảnh PNG.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("Không thể import matplotlib hoặc numpy, bỏ qua bước vẽ biểu đồ.")
        return

    METRICS = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_correctness",
    ]
    METRIC_LABELS = {
        "faithfulness":       "Faithfulness\n(Trung thực)",
        "answer_relevancy":   "Answer Relevancy\n(Liên quan)",
        "context_precision":  "Context Precision\n(Chính xác)",
        "context_recall":     "Context Recall\n(Đầy đủ)",
        "answer_correctness": "Answer Correctness\n(Đúng đắn)",
    }

    available_metrics = [m for m in METRICS if m in df_results.columns]
    avg_scores = df_results[available_metrics].mean()
    plot_scores = np.nan_to_num(avg_scores.values, nan=0.0)

    # 1. Bar Chart & Radar Chart (Side by Side)
    fig = plt.figure(figsize=(15, 6))
    fig.suptitle("KẾT QUẢ ĐÁNH GIÁ CHẤT LƯỢNG CHATBOT (RAGAS)", fontsize=15, fontweight="bold", y=0.98)

    # Bar chart
    ax1 = fig.add_subplot(1, 2, 1)
    colors = [
        "#27ae60" if (not np.isnan(s) and s >= 0.7) else ("#f39c12" if (not np.isnan(s) and s >= 0.5) else "#d35400")
        for s in avg_scores.values
    ]
    bars = ax1.bar(
        [METRIC_LABELS.get(m, m) for m in available_metrics],
        plot_scores,
        color=colors, width=0.5, edgecolor="white", linewidth=1.2
    )
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("Điểm số (Score)", fontsize=11)
    ax1.set_title("Điểm Trung Bình Từng Metric", fontsize=12, fontweight="bold")
    ax1.axhline(y=0.7, color="#27ae60", linestyle="--", alpha=0.6, label="Tốt (>=0.7)")
    ax1.axhline(y=0.5, color="#f39c12", linestyle="--", alpha=0.6, label="Tạm ổn (>=0.5)")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_xticks(range(len(available_metrics)))
    ax1.set_xticklabels([METRIC_LABELS.get(m, m) for m in available_metrics], fontsize=9)
    for bar, score in zip(bars, avg_scores.values):
        height = bar.get_height()
        label_val = f"{score:.3f}" if not np.isnan(score) else "NaN"
        ax1.text(bar.get_x() + bar.get_width() / 2.0, height + 0.02,
                 label_val, ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax1.set_facecolor("#fcfcfc")
    ax1.grid(axis="y", alpha=0.2)

    # Radar chart
    ax2 = fig.add_subplot(1, 2, 2, projection="polar")
    N = len(available_metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values = plot_scores.tolist()
    
    # Close the polygon
    angles_c = angles + angles[:1]
    values_c = values + values[:1]

    ax2.plot(angles_c, values_c, "o-", linewidth=2, color="#2980b9")
    ax2.fill(angles_c, values_c, alpha=0.2, color="#2980b9")
    ax2.set_xticks(angles)
    ax2.set_xticklabels([m.replace("_", "\n").title() for m in available_metrics], fontsize=9)
    ax2.set_ylim(0, 1.0)
    ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax2.set_title("Biểu Đồ Radar Đa Chiều", pad=20, fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "ragas_metrics_chart.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Đã cập nhật biểu đồ: {chart_path}")

    # 2. Heatmap per Sample
    heatmap_data = df_results[available_metrics].values.astype(float)
    question_col = "user_input" if "user_input" in df_results.columns else "question"
    q_labels = [f"Q{i+1}: {str(q)[:30]}..." for i, q in enumerate(df_results.get(question_col, ["Unknown"] * len(df_results)))]

    fig, ax = plt.subplots(figsize=(10, max(4, len(df_results) * 0.5)))
    im = ax.imshow(heatmap_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

    ax.set_xticks(range(len(available_metrics)))
    ax.set_xticklabels(available_metrics, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(df_results)))
    ax.set_yticklabels(q_labels, fontsize=9)
    ax.set_title("Bản Đồ Nhiệt (Heatmap) Điểm Số Từng Mẫu", fontsize=13, fontweight="bold", pad=12)

    for i in range(len(df_results)):
        for j in range(len(available_metrics)):
            val = heatmap_data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8.5, fontweight="bold",
                        color="white" if val < 0.4 else "black")

    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "ragas_heatmap.png")
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ Đã cập nhật Heatmap: {heatmap_path}")

# ──────────────────────────────────────────────────────────────
# Chạy RAGAS Evaluation
# ──────────────────────────────────────────────────────────────
def run_ragas_evaluation(
    rag_results: list[dict],
    output_dir: str = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Chạy RAGAS evaluation trên tập kết quả từ RAG pipeline.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "results")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Chuyển sang Hugging Face Dataset format
    dataset_dict = {
        "question": [r["question"] for r in rag_results],
        "answer": [r["answer"] for r in rag_results],
        "contexts": [r["contexts"] for r in rag_results],
        "ground_truth": [r["ground_truth"] for r in rag_results],
    }
    hf_dataset = Dataset.from_dict(dataset_dict)

    logger.info("Đang khởi tạo LLM và Embeddings cho RAGAS...")
    ragas_llm = get_ragas_llm()
    ragas_embeddings = get_ragas_embeddings()

    # Khởi tạo các metric RAGAS
    metrics = [
        Faithfulness(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]

    logger.info(f"Bắt đầu chấm điểm {len(rag_results)} mẫu với {len(metrics)} metrics...")
    logger.info(f"Metrics: {[m.name for m in metrics]}")

    run_config = RunConfig(max_workers=1, timeout=300)
    result = evaluate(
        dataset=hf_dataset,
        metrics=metrics,
        run_config=run_config,
        raise_exceptions=False,
    )

    df = result.to_pandas()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df["timestamp"] = timestamp

    # Tính điểm trung bình
    metric_cols = [m.name for m in metrics]
    avg_scores = df[metric_cols].mean()

    logger.info("\n" + "=" * 60)
    logger.info("KẾT QUẢ ĐÁNH GIÁ RAGAS TRUNG BÌNH")
    logger.info("=" * 60)
    for col, score in avg_scores.items():
        if pd.isna(score):
            bar = "░" * 20
            logger.info(f"  {col:<25} {bar} NaN (Failed/API Quota Exceeded)")
        else:
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            logger.info(f"  {col:<25} {bar} {score:.4f}")
    logger.info("=" * 60)

    # Lưu kết quả theo dạng timestamp
    csv_path = os.path.join(output_dir, f"ragas_results_{timestamp}.csv")
    json_path = os.path.join(output_dir, f"ragas_results_{timestamp}.json")
    summary_path = os.path.join(output_dir, f"ragas_summary_{timestamp}.json")

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)

    summary = {
        "evaluation_timestamp": timestamp,
        "total_samples": len(rag_results),
        "metrics_used": [m.name for m in metrics],
        "average_scores": avg_scores.to_dict(),
        "overall_score": avg_scores.mean(),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Lưu thêm phiên bản tĩnh "latest" để dashboard dễ truy cập
    latest_json_path = os.path.join(output_dir, "ragas_results_latest.json")
    latest_summary_path = os.path.join(output_dir, "ragas_summary_latest.json")
    
    df.to_json(latest_json_path, orient="records", force_ascii=False, indent=2)
    with open(latest_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        
    logger.info("Đã lưu các kết quả dạng timestamp và dạng 'latest'")

    # Vẽ biểu đồ tĩnh
    generate_plots(df, output_dir)

    return df, summary



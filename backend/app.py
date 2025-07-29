from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from embedder import embed_text
from pinecone_utils import upsert_embedding, query_embedding, delete_all_chunks, index, namespace
from llm_qa import analyze_fit, assess_resume
import pdfplumber
import uuid
from nltk.tokenize import sent_tokenize
import nltk
from dotenv import load_dotenv
import os
import io
import json
import logging
from typing import List, Dict, Any, Optional
from werkzeug.exceptions import RequestEntityTooLarge
import time
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

load_dotenv()

app = Flask(__name__)
CORS(app)

MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
MAX_CHUNK_LENGTH = 500
MIN_CHUNK_LENGTH = 50
MIN_SIMILARITY_SCORE = 0.15  # Lowered from 0.2 to get more results
MAX_CONTEXT_CHUNKS = 15  # Increased from 10
SUPPORTED_EXTENSIONS = {'.pdf'}
EMBEDDING_DIM = 384

app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        logger.info(f"{func.__name__} executed in {time.time() - start_time:.2f} seconds")
        return result
    return wrapper

def validate_file(file) -> tuple[bool, str]:
    if not file or not file.filename:
        return False, "No file provided"
    if not any(file.filename.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
        return False, f"Unsupported file type. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
    return True, ""

def split_into_chunks(text: str, max_length: int = MAX_CHUNK_LENGTH) -> List[str]:
    if not text or not text.strip():
        return []
    try:
        sentences = sent_tokenize(text)
    except Exception:
        sentences = [s.strip() for s in text.replace('\n', '. ').split('.') if s.strip()]
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current_chunk) + len(sentence) + 1 > max_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk = f"{current_chunk} {sentence}" if current_chunk else sentence
    if current_chunk and current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks

def is_low_quality_chunk(chunk: str) -> bool:
    if len(chunk) < MIN_CHUNK_LENGTH:
        return True
    lower_chunk = chunk.lower()
    contact_keywords = ["email", "linkedin", "github", "phone", "contact", "tel:", "mailto:", "@"]
    contact_count = sum(1 for kw in contact_keywords if kw in lower_chunk)
    if contact_count >= 3:
        return True
    if len(chunk.strip().replace(' ', '').replace('\n', '').replace('\t', '')) < MIN_CHUNK_LENGTH // 2:
        return True
    for char in ['-', '=', '_', '*', '.']:
        if chunk.count(char) > len(chunk) * 0.3:
            return True
    return False

def convert_chat_history(history: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    if not history:
        return []
    converted = []
    for i, item in enumerate(history):
        if not isinstance(item, dict):
            continue
        role = "user" if i % 2 == 0 else "assistant"
        content = item.get("query") if role == "user" else item.get("response")
        if content and isinstance(content, str):
            converted.append({"role": role, "content": content})
    return converted

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(error):
    return jsonify({"error": f"File too large. Maximum size: {MAX_FILE_SIZE // 1024 // 1024}MB"}), 413

@app.errorhandler(Exception)
def handle_unexpected_error(error):
    logger.error(f"Unexpected error: {str(error)}", exc_info=True)
    return jsonify({"error": "An unexpected error occurred"}), 500

@app.route('/upload-pdf', methods=['POST'])
@timer_decorator
def upload_pdf():
    try:
        files = request.files.getlist("files")
        if not files:
            return jsonify({"error": "No files provided in request"}), 400
        all_results = []
        total_chunks_processed = 0
        for file in files:
            is_valid, error_msg = validate_file(file)
            if not is_valid:
                all_results.append({"file": file.filename, "error": error_msg, "chunks": []})
                continue
            with pdfplumber.open(file) as pdf:
                full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            if not full_text.strip():
                all_results.append({"file": file.filename, "error": "No text could be extracted from PDF", "chunks": []})
                continue
            chunks = split_into_chunks(full_text, MAX_CHUNK_LENGTH)
            high_quality_chunks = [c for c in chunks if not is_low_quality_chunk(c)]
            successful_chunks = []
            failed_chunks = 0
            for i, chunk in enumerate(high_quality_chunks):
                chunk_id = f"{file.filename}_chunk_{i}_{uuid.uuid4().hex[:8]}"
                try:
                    embedding = embed_text(chunk)
                    metadata = {
                        "chunk_index": i,
                        "source": file.filename,
                        "preview": chunk[:300],
                        "full_text": chunk,
                        "uploaded_at": time.time()
                    }
                    upsert_embedding(id=chunk_id, embedding=embedding, metadata=metadata)
                    successful_chunks.append({"id": chunk_id, "metadata": {k: v for k, v in metadata.items() if k != "full_text"}})
                except Exception as e:
                    logger.error(f"Failed to process chunk {chunk_id}: {str(e)}")
                    failed_chunks += 1
            total_chunks_processed += len(successful_chunks)
            all_results.append({
                "file": file.filename,
                "message": f"Successfully processed {len(successful_chunks)} chunks from '{file.filename}'",
                "chunks": successful_chunks,
                "total_chunks_generated": len(chunks),
                "quality_filtered": len(high_quality_chunks),
                "successful_uploads": len(successful_chunks),
                "failed_uploads": failed_chunks
            })
        if not any(r.get("chunks") for r in all_results):
            return jsonify({"error": "No chunks were successfully processed from any files", "details": all_results}), 400
        return jsonify({
            "message": f"Processing complete. {total_chunks_processed} total chunks uploaded.",
            "results": all_results,
            "summary": {
                "total_files": len(files),
                "successful_files": len([r for r in all_results if r.get("chunks")]),
                "total_chunks": total_chunks_processed
            }
        })
    except Exception as e:
        logger.error(f"Unexpected error in upload_pdf: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to process upload request"}), 500

@app.route('/qa', methods=['POST'])
@timer_decorator
def jd_match_route():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        jd_text = data.get("query", "").strip()
        top_k = min(max(data.get("top_k", 15), 1), 50)
        chat_history = data.get("chat_history")
        uploaded_files = data.get("uploaded_files", [])
        if not jd_text:
            return jsonify({"error": "Job description is required"}), 400
        jd_embedding = embed_text(jd_text)
        results = query_embedding(jd_embedding, top_k=top_k)
        filtered_results = []
        for r in results:
            score = r.get("score", 0)
            source = r.get("metadata", {}).get("source", "")
            if score >= MIN_SIMILARITY_SCORE:
                if not uploaded_files or source in uploaded_files:
                    filtered_results.append(r)
        if not filtered_results:
            return jsonify({
                "query": jd_text,
                "summary": {
                    "Summary": "No relevant resume matches found for this job description.",
                    "Verdict": "No Fit",
                    "Score": 0,
                    "Relevant Skills": []
                },
                "chunks_used": [],
                "metadata": {
                    "total_results": len(results),
                    "filtered_results": 0,
                    "min_score_threshold": MIN_SIMILARITY_SCORE
                }
            })
        context_chunks = []
        seen_previews = set()
        for match in filtered_results:
            meta = match.get("metadata", {})
            preview = meta.get("preview", "")
            if not preview or preview in seen_previews:
                continue
            seen_previews.add(preview)
            context_chunks.append({
                "id": match.get("id", ""),
                "score": round(match.get("score", 0), 3),
                "preview": preview,
                "source": meta.get("source", "Unknown"),
                "match_snippet": preview
            })
            if len(context_chunks) >= MAX_CONTEXT_CHUNKS:
                break
        chat_history_converted = convert_chat_history(chat_history)
        summary = analyze_fit(jd_text, context_chunks, chat_history_converted)
        return jsonify({
            "query": jd_text,
            "summary": summary,
            "chunks_used": context_chunks,
            "metadata": {
                "total_results": len(results),
                "filtered_results": len(filtered_results),
                "context_chunks_used": len(context_chunks),
                "min_score_threshold": MIN_SIMILARITY_SCORE,
                "average_similarity": round(sum(c["score"] for c in context_chunks) / len(context_chunks), 3) if context_chunks else 0
            }
        })
    except Exception as e:
        logger.error(f"Unexpected error in jd_match_route: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to process job description analysis"}), 500

@app.route('/assess-resume', methods=['POST'])
@timer_decorator
def assess_resume_route():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        jd_text = data.get("job_description", "").strip()
        resume_file = data.get("resume_file", "").strip()
        top_k = min(max(data.get("top_k", 30), 1), 50)  # Increased default top_k
        chat_history = data.get("chat_history")
        
        if not jd_text:
            return jsonify({"error": "Job description is required"}), 400
        if not resume_file:
            return jsonify({"error": "Resume file must be specified"}), 400
        
        logger.info(f"Assessing resume: {resume_file} against job description")
        
        # Get embeddings and query with higher top_k to get more results
        jd_embedding = embed_text(jd_text)
        results = query_embedding(jd_embedding, top_k=top_k * 3)  # Get more results initially
        
        logger.info(f"Retrieved {len(results)} total results from vector search")
        
        # Filter for the specific resume file with more lenient scoring
        filtered_results = []
        all_sources = set()  # Track all sources found
        for r in results:
            score = r.get("score", 0)
            source = r.get("metadata", {}).get("source", "")
            all_sources.add(source)
            
            # More lenient filtering - lower threshold and exact file match
            if source == resume_file and score >= 0.1:  # Even lower threshold
                filtered_results.append(r)
        
        logger.info(f"Found sources: {all_sources}")
        logger.info(f"Filtered results for {resume_file}: {len(filtered_results)} chunks")
        
        # If still no results, try alternative approach
        if not filtered_results:
            logger.warning(f"No chunks found for {resume_file}, trying alternative search")
            
            # Try searching with just the resume filename without extension
            resume_name_variants = [
                resume_file,
                resume_file.replace('.pdf', ''),
                resume_file.replace('.PDF', ''),
                resume_file.split('.')[0] if '.' in resume_file else resume_file
            ]
            
            # Get all chunks and filter by any variant of the filename
            all_results = query_embedding([0.0] * EMBEDDING_DIM, top_k=1000)  # Get all chunks
            for r in all_results:
                source = r.get("metadata", {}).get("source", "")
                if any(variant in source for variant in resume_name_variants):
                    filtered_results.append(r)
            
            logger.info(f"Alternative search found {len(filtered_results)} chunks")
        
        if not filtered_results:
            available_files = list(all_sources) if all_sources else ["No files found"]
            return jsonify({
                "error": f"No content found for resume '{resume_file}'. Available files: {available_files}",
                "details": {
                    "requested_file": resume_file,
                    "available_files": available_files,
                    "total_results": len(results),
                    "filtered_results": 0,
                    "min_score_threshold": 0.1
                }
            }), 400
        
        # Build context chunks for assessment
        context_chunks = []
        seen_previews = set()
        for match in filtered_results:
            meta = match.get("metadata", {})
            preview = meta.get("preview", "")
            if not preview or preview in seen_previews:
                continue
            seen_previews.add(preview)
            context_chunks.append({
                "id": match.get("id", ""),
                "score": round(match.get("score", 0), 3),
                "preview": preview,
                "source": meta.get("source", "Unknown")
            })
            if len(context_chunks) >= MAX_CONTEXT_CHUNKS:
                break
        
        logger.info(f"Using {len(context_chunks)} context chunks for assessment")
        
        # Convert chat history and perform assessment
        chat_history_converted = convert_chat_history(chat_history)
        assessment_result = assess_resume(jd_text, context_chunks, chat_history_converted)
        
        return jsonify({
            "job_description": jd_text,
            "resume_file": resume_file,
            "assessment": assessment_result,
            "chunks_used": context_chunks,
            "metadata": {
                "total_results": len(results),
                "filtered_results": len(filtered_results),
                "context_chunks_used": len(context_chunks),
                "average_similarity": round(sum(c["score"] for c in context_chunks) / len(context_chunks), 3) if context_chunks else 0
            }
        })
    
    except Exception as e:
        logger.error(f"Unexpected error in assess_resume_route: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to process resume assessment"}), 500

@app.route('/clear-index', methods=['POST'])
def clear_index():
    try:
        delete_all_chunks()
        logger.info("Successfully cleared all data from Pinecone index")
        return jsonify({"message": "All data cleared from Pinecone index"}), 200
    except Exception as e:
        logger.error(f"Failed to clear index: {str(e)}")
        return jsonify({"error": f"Failed to clear index: {str(e)}"}), 500

@app.route('/count-vectors', methods=['GET'])
def count_vectors():
    try:
        stats = index.describe_index_stats(namespace=namespace)
        count = stats.get('namespaces', {}).get(namespace, {}).get('vector_count', 0)
        total_count = stats.get('total_vector_count', 0)
        return jsonify({
            "vector_count": count,
            "total_count": total_count,
            "namespace": namespace
        })
    except Exception as e:
        logger.error(f"Failed to get vector count: {str(e)}")
        return jsonify({"error": f"Failed to retrieve vector count: {str(e)}"}), 500

@app.route('/list-uploaded-files', methods=['GET'])
def list_uploaded_files():
    try:
        dummy_embedding = [0.0] * EMBEDDING_DIM
        sample_results = query_embedding(dummy_embedding, top_k=1000)
        sources = set()
        for result in sample_results:
            metadata = result.get("metadata", {})
            source = metadata.get("source", "")
            if source:
                sources.add(source)
        sources_list = sorted(list(sources))
        logger.info(f"Found {len(sources_list)} uploaded files.")
        return jsonify({
            "uploaded_files": sources_list,
            "total_files": len(sources_list)
        })
    except Exception as e:
        logger.error(f"Failed to list uploaded files: {str(e)}", exc_info=True)
        return jsonify({"error": f"Failed to retrieve file list: {str(e)}"}), 500

@app.route('/export', methods=['POST'])
def export_analysis():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided for export"}), 400
        export_content = {
            "timestamp": time.time(),
            "analysis_type": data.get("analysis_type", "unknown"),
            "job_description": data.get("job_description", ""),
            "summary": data.get("summary", {}),
            "assessment": data.get("assessment", {}),
            "chunks_used": data.get("chunks_used", []),
            "metadata": data.get("metadata", {}),
            "export_version": "3.0"
        }
        json_str = json.dumps(export_content, indent=2, ensure_ascii=False)
        buffer = io.BytesIO()
        buffer.write(json_str.encode('utf-8'))
        buffer.seek(0)
        analysis_type = data.get("analysis_type", "analysis")
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f"resume_{analysis_type}_{int(time.time())}.json",
            mimetype="application/json"
        )
    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        return jsonify({"error": f"Export failed: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    try:
        stats = index.describe_index_stats(namespace=namespace)
        vector_count = stats.get('namespaces', {}).get(namespace, {}).get('vector_count', 0)
        return jsonify({
            "status": "healthy",
            "timestamp": time.time(),
            "vector_count": vector_count,
            "version": "3.0",
            "features": [
                "resume_analysis",
                "resume_assessment",
                "weakness_analysis",
                "improvement_recommendations"
            ]
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }), 500

if __name__ == "__main__":
    logger.info("Starting Flask application with resume fit analysis and assessment features...")
    app.run(debug=True, host='0.0.0.0', port=5000)
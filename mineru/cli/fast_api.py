import uuid
import os
import uvicorn
import click
import zipfile
from pathlib import Path
from glob import glob
from io import BytesIO
from urllib.parse import quote
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from typing import List, Optional
from loguru import logger
from base64 import b64encode

from mineru.cli.common import aio_do_parse, read_fn, pdf_suffixes, image_suffixes
from mineru.utils.cli_parser import arg_parse
from mineru.version import __version__

app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)

def encode_image(image_path: str) -> str:
    """Encode image using base64"""
    with open(image_path, "rb") as f:
        return b64encode(f.read()).decode()


def get_infer_result(file_suffix_identifier: str, pdf_name: str, parse_dir: str) -> Optional[str]:
    """从结果文件中读取推理结果"""
    result_file_path = os.path.join(parse_dir, f"{pdf_name}{file_suffix_identifier}")
    if os.path.exists(result_file_path):
        with open(result_file_path, "r", encoding="utf-8") as fp:
            return fp.read()
    return None


def create_zip_archive(result_dict: dict, pdf_file_names: list, parse_dirs: dict, return_options: dict) -> BytesIO:
    """
    根据勾选的输出选项创建ZIP压缩包
    
    Args:
        result_dict: 解析结果字典
        pdf_file_names: PDF文件名列表
        parse_dirs: 解析目录字典
        return_options: 返回选项字典
    
    Returns:
        BytesIO: ZIP文件的字节流
    """
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zip_file:
        for pdf_name in pdf_file_names:
            parse_dir = parse_dirs.get(pdf_name)
            if not parse_dir or not os.path.exists(parse_dir):
                continue
            
            # 为每个文件创建一个文件夹，使用安全的文件名
            safe_name = "".join(c for c in pdf_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            folder_name = f"{safe_name}/"
            
            # 添加Markdown文件
            if return_options.get('return_md', False):
                md_content = get_infer_result(".md", pdf_name, parse_dir)
                if md_content:
                    zip_file.writestr(
                        f"{folder_name}{safe_name}.md", 
                        md_content.encode('utf-8'),
                        compress_type=zipfile.ZIP_DEFLATED
                    )
            
            # 添加中间JSON文件
            if return_options.get('return_middle_json', False):
                middle_json = get_infer_result("_middle.json", pdf_name, parse_dir)
                if middle_json:
                    zip_file.writestr(
                        f"{folder_name}{safe_name}_middle.json", 
                        middle_json.encode('utf-8'),
                        compress_type=zipfile.ZIP_DEFLATED
                    )
            
            # 添加模型输出文件
            if return_options.get('return_model_output', False):
                model_output = get_infer_result("_model.json", pdf_name, parse_dir)
                if not model_output:
                    model_output = get_infer_result("_model_output.txt", pdf_name, parse_dir)
                if model_output:
                    ext = ".json" if "_model.json" in parse_dir else ".txt"
                    zip_file.writestr(
                        f"{folder_name}{safe_name}_model_output{ext}", 
                        model_output.encode('utf-8'),
                        compress_type=zipfile.ZIP_DEFLATED
                    )
            
            # 添加内容列表文件
            if return_options.get('return_content_list', False):
                content_list = get_infer_result("_content_list.json", pdf_name, parse_dir)
                if content_list:
                    zip_file.writestr(
                        f"{folder_name}{safe_name}_content_list.json", 
                        content_list.encode('utf-8'),
                        compress_type=zipfile.ZIP_DEFLATED
                    )
            
            # 添加图片文件
            if return_options.get('return_images', False):
                images_dir = os.path.join(parse_dir, "images")
                if os.path.exists(images_dir):
                    image_paths = glob(f"{images_dir}/*.jpg") + glob(f"{images_dir}/*.png")
                    for image_path in image_paths:
                        image_name = os.path.basename(image_path)
                        zip_file.write(image_path, f"{folder_name}images/{image_name}")
    
    zip_buffer.seek(0)
    return zip_buffer


@app.get("/", response_class=HTMLResponse)
async def upload_form():
    """返回文件上传的HTML页面"""
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MinerU 文件解析服务</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            .container {
                background: white;
                border-radius: 15px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
                padding: 40px;
                max-width: 800px;
                width: 100%;
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
            }
            .header h1 {
                color: #333;
                font-size: 2.5em;
                margin-bottom: 10px;
            }
            .header p {
                color: #666;
                font-size: 1.1em;
            }
            .form-group {
                margin-bottom: 20px;
            }
            .form-group label {
                display: block;
                margin-bottom: 8px;
                font-weight: 600;
                color: #333;
            }
            .form-group input,
            .form-group select {
                width: 100%;
                padding: 12px;
                border: 2px solid #e1e5e9;
                border-radius: 8px;
                font-size: 16px;
                transition: border-color 0.3s ease;
            }
            .form-group input:focus,
            .form-group select:focus {
                outline: none;
                border-color: #667eea;
            }
            .file-upload {
                border: 2px dashed #e1e5e9;
                border-radius: 8px;
                padding: 40px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            .file-upload:hover {
                border-color: #667eea;
                background-color: #f8f9ff;
            }
            .file-upload.dragover {
                border-color: #667eea;
                background-color: #f0f2ff;
            }
            .file-upload-text {
                color: #666;
                font-size: 16px;
                margin-bottom: 10px;
            }
            .file-upload-hint {
                color: #999;
                font-size: 14px;
            }
            .checkbox-group {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-top: 10px;
            }
            .checkbox-item {
                display: flex;
                align-items: center;
            }
            .checkbox-item input {
                margin-right: 8px;
                width: auto;
            }
            .submit-btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px 30px;
                border: none;
                border-radius: 8px;
                font-size: 18px;
                font-weight: 600;
                cursor: pointer;
                width: 100%;
                transition: transform 0.3s ease;
            }
            .submit-btn:hover {
                transform: translateY(-2px);
            }
            .submit-btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            .loading {
                display: none;
                text-align: center;
                margin-top: 20px;
            }
            .spinner {
                border: 3px solid #f3f3f3;
                border-top: 3px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto 10px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .result {
                margin-top: 30px;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 8px;
                border-left: 4px solid #667eea;
                display: none;
            }
            .result pre {
                white-space: pre-wrap;
                word-wrap: break-word;
                background: #fff;
                padding: 15px;
                border-radius: 5px;
                max-height: 400px;
                overflow-y: auto;
            }
            .selected-files {
                margin-top: 10px;
                padding: 10px;
                background-color: #f8f9fa;
                border-radius: 5px;
                display: none;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>MinerU 文件解析</h1>
                <p>上传PDF或图片文件进行智能解析</p>
            </div>
            
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="form-group">
                    <label>选择文件 (支持PDF、图片格式):</label>
                    <div class="file-upload" id="fileUpload">
                        <div class="file-upload-text">点击选择文件或拖拽文件到此处</div>
                        <div class="file-upload-hint">支持 PDF, JPG, PNG, JPEG 格式</div>
                        <input type="file" id="files" name="files" multiple accept=".pdf,.jpg,.jpeg,.png" style="display: none;">
                    </div>
                    <div class="selected-files" id="selectedFiles"></div>
                </div>

                <div class="form-group">
                    <label for="outputDir">输出目录:</label>
                    <input type="text" id="outputDir" name="output_dir" value="./output" placeholder="输出目录路径">
                </div>

                <div class="form-group">
                    <label for="backend">后端模式:</label>
                    <select id="backend" name="backend">
                        <option value="pipeline">Pipeline</option>
                        <option value="vlm">VLM</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="parseMethod">解析方法:</label>
                    <select id="parseMethod" name="parse_method">
                        <option value="auto">自动</option>
                        <option value="ocr">OCR</option>
                        <option value="txt">文本</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="langList">语言:</label>
                    <select id="langList" name="lang_list">
                        <option value="ch">中文</option>
                        <option value="en">英文</option>
                        <option value="ja">日文</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>输出选项:</label>
                    <div class="checkbox-group">
                        <div class="checkbox-item">
                            <input type="checkbox" id="returnMd" name="return_md" checked>
                            <label for="returnMd">返回Markdown</label>
                        </div>
                        <div class="checkbox-item">
                            <input type="checkbox" id="returnMiddleJson" name="return_middle_json">
                            <label for="returnMiddleJson">返回中间JSON</label>
                        </div>
                        <div class="checkbox-item">
                            <input type="checkbox" id="returnModelOutput" name="return_model_output">
                            <label for="returnModelOutput">返回模型输出</label>
                        </div>
                        <div class="checkbox-item">
                            <input type="checkbox" id="returnContentList" name="return_content_list">
                            <label for="returnContentList">返回内容列表</label>
                        </div>
                        <div class="checkbox-item">
                            <input type="checkbox" id="returnImages" name="return_images">
                            <label for="returnImages">返回图片</label>
                        </div>
                        <div class="checkbox-item">
                            <input type="checkbox" id="formulaEnable" name="formula_enable" checked>
                            <label for="formulaEnable">启用公式识别</label>
                        </div>
                        <div class="checkbox-item">
                            <input type="checkbox" id="tableEnable" name="table_enable" checked>
                            <label for="tableEnable">启用表格识别</label>
                        </div>
                        <div class="checkbox-item">
                            <input type="checkbox" id="returnZip" name="return_zip">
                            <label for="returnZip">下载ZIP压缩包</label>
                        </div>
                    </div>
                </div>

                <button type="submit" class="submit-btn" id="submitBtn">开始解析</button>
            </form>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>正在处理文件，请稍候...</p>
            </div>

            <div class="result" id="result">
                <h3>解析结果:</h3>
                <pre id="resultContent"></pre>
            </div>
        </div>

        <script>
            const fileUpload = document.getElementById('fileUpload');
            const fileInput = document.getElementById('files');
            const selectedFiles = document.getElementById('selectedFiles');
            const form = document.getElementById('uploadForm');
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            const resultContent = document.getElementById('resultContent');
            const submitBtn = document.getElementById('submitBtn');

            // 文件上传区域点击事件
            fileUpload.addEventListener('click', () => {
                fileInput.click();
            });

            // 拖拽事件
            fileUpload.addEventListener('dragover', (e) => {
                e.preventDefault();
                fileUpload.classList.add('dragover');
            });

            fileUpload.addEventListener('dragleave', () => {
                fileUpload.classList.remove('dragover');
            });

            fileUpload.addEventListener('drop', (e) => {
                e.preventDefault();
                fileUpload.classList.remove('dragover');
                fileInput.files = e.dataTransfer.files;
                displaySelectedFiles();
            });

            // 文件选择事件
            fileInput.addEventListener('change', displaySelectedFiles);

            function displaySelectedFiles() {
                const files = fileInput.files;
                if (files.length > 0) {
                    let fileList = '<strong>已选择文件:</strong><br>';
                    for (let i = 0; i < files.length; i++) {
                        fileList += `${i + 1}. ${files[i].name} (${(files[i].size / 1024 / 1024).toFixed(2)} MB)<br>`;
                    }
                    selectedFiles.innerHTML = fileList;
                    selectedFiles.style.display = 'block';
                } else {
                    selectedFiles.style.display = 'none';
                }
            }

            // 表单提交事件
            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                console.log('Form submission started');
                
                if (fileInput.files.length === 0) {
                    alert('请选择要解析的文件！');
                    return;
                }

                console.log('Files selected:', fileInput.files.length);
                const formData = new FormData();
                
                // 添加文件
                for (let file of fileInput.files) {
                    formData.append('files', file);
                }
                
                // 添加其他参数
                formData.append('output_dir', document.getElementById('outputDir').value);
                formData.append('backend', document.getElementById('backend').value);
                formData.append('parse_method', document.getElementById('parseMethod').value);
                formData.append('lang_list', document.getElementById('langList').value);
                
                // 添加布尔参数
                formData.append('return_md', document.getElementById('returnMd').checked);
                formData.append('return_middle_json', document.getElementById('returnMiddleJson').checked);
                formData.append('return_model_output', document.getElementById('returnModelOutput').checked);
                formData.append('return_content_list', document.getElementById('returnContentList').checked);
                formData.append('return_images', document.getElementById('returnImages').checked);
                formData.append('formula_enable', document.getElementById('formulaEnable').checked);
                formData.append('table_enable', document.getElementById('tableEnable').checked);
                formData.append('return_zip', document.getElementById('returnZip').checked);

                console.log('FormData prepared, sending request...');
                
                // 显示加载状态
                loading.style.display = 'block';
                result.style.display = 'none';
                submitBtn.disabled = true;

                try {
                    console.log('Sending POST request to /file_parse');
                    const response = await fetch('/file_parse', {
                        method: 'POST',
                        body: formData
                    });

                    console.log('Response received:', response.status);
                    
                    if (response.ok) {
                        const contentType = response.headers.get('Content-Type');
                        
                        // 检查是否返回ZIP文件
                        if (contentType && contentType.includes('application/zip')) {
                            // 处理ZIP文件下载
                            const blob = await response.blob();
                            const url = window.URL.createObjectURL(blob);
                            
                            // 从Content-Disposition头中获取文件名
                            const disposition = response.headers.get('Content-Disposition');
                            let filename = 'parsed_files.zip';
                            if (disposition && disposition.includes('filename=')) {
                                const matches = disposition.match(/filename=([^;]+)/);
                                if (matches && matches[1]) {
                                    filename = matches[1];
                                }
                            }
                            
                            // 创建下载链接并触发下载
                            const a = document.createElement('a');
                            a.href = url;
                            a.download = filename;
                            document.body.appendChild(a);
                            a.click();
                            document.body.removeChild(a);
                            window.URL.revokeObjectURL(url);
                            
                            resultContent.textContent = `ZIP文件已下载: ${filename}`;
                            result.style.display = 'block';
                        } else {
                            // 处理JSON响应
                            const data = await response.json();
                            console.log('Response data:', data);
                            
                            resultContent.textContent = JSON.stringify(data, null, 2);
                            result.style.display = 'block';
                        }
                    } else {
                        const data = await response.json();
                        resultContent.textContent = '错误: ' + (data.error || '解析失败');
                        result.style.display = 'block';
                    }
                } catch (error) {
                    console.error('Request failed:', error);
                    resultContent.textContent = '网络错误: ' + error.message;
                    result.style.display = 'block';
                } finally {
                    loading.style.display = 'none';
                    submitBtn.disabled = false;
                    console.log('Request completed');
                }
            });
            
            // 添加点击事件监听器到按钮，用于调试
            submitBtn.addEventListener('click', function(e) {
                console.log('Submit button clicked');
            });
        </script>
    </body>
    </html>
    """
    return html_content


@app.post(path="/file_parse",)
async def parse_pdf(
        files: List[UploadFile] = File(...),
        output_dir: str = Form("./output"),
        lang_list: List[str] = Form(["ch"]),
        backend: str = Form("pipeline"),
        parse_method: str = Form("auto"),
        formula_enable: bool = Form(True),
        table_enable: bool = Form(True),
        server_url: Optional[str] = Form(None),
        return_md: bool = Form(True),
        return_middle_json: bool = Form(False),
        return_model_output: bool = Form(False),
        return_content_list: bool = Form(False),
        return_images: bool = Form(False),
        return_zip: bool = Form(False),
        start_page_id: int = Form(0),
        end_page_id: int = Form(99999),
):

    # 获取命令行配置参数
    config = getattr(app.state, "config", {})

    try:
        # 创建唯一的输出目录
        unique_dir = os.path.join(output_dir, str(uuid.uuid4()))
        os.makedirs(unique_dir, exist_ok=True)

        # 处理上传的PDF文件
        pdf_file_names = []
        pdf_bytes_list = []

        for file in files:
            content = await file.read()
            file_path = Path(file.filename)

            # 如果是图像文件或PDF，使用read_fn处理
            if file_path.suffix.lower() in pdf_suffixes + image_suffixes:
                # 创建临时文件以便使用read_fn
                temp_path = Path(unique_dir) / file_path.name
                with open(temp_path, "wb") as f:
                    f.write(content)

                try:
                    pdf_bytes = read_fn(temp_path)
                    pdf_bytes_list.append(pdf_bytes)
                    pdf_file_names.append(file_path.stem)
                    os.remove(temp_path)  # 删除临时文件
                except Exception as e:
                    return JSONResponse(
                        status_code=400,
                        content={"error": f"Failed to load file: {str(e)}"}
                    )
            else:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Unsupported file type: {file_path.suffix}"}
                )


        # 设置语言列表，确保与文件数量一致
        actual_lang_list = lang_list
        if len(actual_lang_list) != len(pdf_file_names):
            # 如果语言列表长度不匹配，使用第一个语言或默认"ch"
            actual_lang_list = [actual_lang_list[0] if actual_lang_list else "ch"] * len(pdf_file_names)

        # 调用异步处理函数
        await aio_do_parse(
            output_dir=unique_dir,
            pdf_file_names=pdf_file_names,
            pdf_bytes_list=pdf_bytes_list,
            p_lang_list=actual_lang_list,
            backend=backend,
            parse_method=parse_method,
            formula_enable=formula_enable,
            table_enable=table_enable,
            server_url=server_url,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_md=return_md,
            f_dump_middle_json=return_middle_json,
            f_dump_model_output=return_model_output,
            f_dump_orig_pdf=False,
            f_dump_content_list=return_content_list,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
            **config
        )

        # 构建结果路径和解析目录字典
        result_dict = {}
        parse_dirs = {}
        
        for pdf_name in pdf_file_names:
            result_dict[pdf_name] = {}
            data = result_dict[pdf_name]

            if backend.startswith("pipeline"):
                parse_dir = os.path.join(unique_dir, pdf_name, parse_method)
            else:
                parse_dir = os.path.join(unique_dir, pdf_name, "vlm")
            
            parse_dirs[pdf_name] = parse_dir

            if os.path.exists(parse_dir):
                if return_md:
                    data["md_content"] = get_infer_result(".md", pdf_name, parse_dir)
                if return_middle_json:
                    data["middle_json"] = get_infer_result("_middle.json", pdf_name, parse_dir)
                if return_model_output:
                    if backend.startswith("pipeline"):
                        data["model_output"] = get_infer_result("_model.json", pdf_name, parse_dir)
                    else:
                        data["model_output"] = get_infer_result("_model_output.txt", pdf_name, parse_dir)
                if return_content_list:
                    data["content_list"] = get_infer_result("_content_list.json", pdf_name, parse_dir)
                if return_images:
                    image_paths = glob(f"{parse_dir}/images/*.jpg")
                    data["images"] = {
                        os.path.basename(
                            image_path
                        ): f"data:image/jpeg;base64,{encode_image(image_path)}"
                        for image_path in image_paths
                    }
        
        # 如果用户勾选了下载ZIP压缩包
        if return_zip:
            return_options = {
                'return_md': return_md,
                'return_middle_json': return_middle_json,
                'return_model_output': return_model_output,
                'return_content_list': return_content_list,
                'return_images': return_images
            }
            
            zip_buffer = create_zip_archive(result_dict, pdf_file_names, parse_dirs, return_options)
            
            # 生成ZIP文件名，处理中文字符
            if len(pdf_file_names) == 1:
                safe_filename = "".join(c for c in pdf_file_names[0] if c.isalnum() or c in (' ', '-', '_')).rstrip()
                zip_filename = f"{safe_filename}_parsed.zip"
            else:
                zip_filename = "parsed_files.zip"
            
            # 对文件名进行URL编码以支持中文
            encoded_filename = quote(zip_filename, safe='')
            
            return StreamingResponse(
                BytesIO(zip_buffer.read()),
                media_type="application/zip",
                headers={
                    "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}",
                    "Content-Type": "application/zip; charset=utf-8"
                }
            )
        else:
            return JSONResponse(
                status_code=200,
                content={
                    "backend": backend,
                    "version": __version__,
                    "results": result_dict
                }
            )
    except Exception as e:
        logger.exception(e)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process file: {str(e)}"}
        )


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.pass_context
@click.option('--host', default='127.0.0.1', help='Server host (default: 127.0.0.1)')
@click.option('--port', default=8000, type=int, help='Server port (default: 8000)')
@click.option('--reload', is_flag=True, help='Enable auto-reload (development mode)')
def main(ctx, host, port, reload, **kwargs):

    kwargs.update(arg_parse(ctx))

    # 将配置参数存储到应用状态中
    app.state.config = kwargs

    """启动MinerU FastAPI服务器的命令行入口"""
    print(f"Start MinerU FastAPI Service: http://{host}:{port}")
    print("The API documentation can be accessed at the following address:")
    print(f"- Swagger UI: http://{host}:{port}/docs")
    print(f"- ReDoc: http://{host}:{port}/redoc")

    uvicorn.run(
        "mineru.cli.fast_api:app",
        host=host,
        port=port,
        reload=reload
    )


if __name__ == "__main__":
    main()
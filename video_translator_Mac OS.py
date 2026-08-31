# ==================== video_translator_portable.py ====================
# 功能：视频双语字幕生成器（Whisper + 机器翻译 + 字幕烧录）
# 兼容：Windows / macOS / Linux（便携版优先使用同目录 ffmpeg）
# 修复：输出目录自动创建、路径智能识别、错误信息清晰化
# =====================================================================

import os
import sys
import json
import queue
import shutil
import tempfile
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# -------------------- 便携版路径配置（必须在最顶部）--------------------
def get_base_dir():
    """获取程序根目录：打包后指向 exe/app 所在文件夹，开发时指向脚本所在文件夹"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = get_base_dir()
MODELS_DIR = os.path.join(BASE_DIR, "models")

# 如果同目录存在 models 文件夹，自动切换为便携模式
if os.path.exists(MODELS_DIR):
    os.environ["WHISPER_CACHE_DIR"] = os.path.join(MODELS_DIR, "whisper")
    os.environ["HF_HOME"] = os.path.join(MODELS_DIR, "huggingface")

# Windows 便携版强制离线；macOS/Linux 允许首次自动下载模型
if sys.platform == "win32":
    os.environ["HF_HUB_OFFLINE"] = "1"

# 优先使用同目录的 ffmpeg / ffprobe（绿色版关键）
exe_suffix = ".exe" if sys.platform == "win32" else ""
local_ffmpeg = os.path.join(BASE_DIR, f"ffmpeg{exe_suffix}")
local_ffprobe = os.path.join(BASE_DIR, f"ffprobe{exe_suffix}")
if os.path.exists(local_ffmpeg):
    os.environ["IMAGEIO_FFMPEG_EXE"] = local_ffmpeg

FFMPEG = local_ffmpeg if os.path.exists(local_ffmpeg) else "ffmpeg"
FFPROBE = local_ffprobe if os.path.exists(local_ffprobe) else "ffprobe"
# ----------------------------------------------------------------------

import whisper
from moviepy.editor import VideoFileClip
from transformers import MarianMTModel, MarianTokenizer

# ========================= 核心功能函数 ================================

def extract_audio(video_path, audio_path):
    """使用 moviepy 提取音频为 MP3"""
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, codec='mp3', verbose=False, logger=None)
        video.close()
        return True
    except Exception as e:
        raise RuntimeError(f"提取音频失败: {e}")

def transcribe_audio(audio_path, model_name="base"):
    """使用 Whisper 进行语音识别，返回时间片段"""
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path, word_timestamps=True, language='en')
    return result['segments']

def translate_text(text, model_name='Helsinki-NLP/opus-mt-en-zh'):
    """使用 MarianMT 进行英译中（支持离线/在线）"""
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name, local_files_only=False)
        model = MarianMTModel.from_pretrained(model_name, local_files_only=False)
        batch = tokenizer([text], return_tensors="pt", truncation=True, padding=True)
        generated_ids = model.generate(**batch)
        translated = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return translated
    except Exception as e:
        return f"[翻译错误: {e}]"

def create_subtitles(segments, output_srt_path, translate=False):
    """生成双语 SRT 字幕文件"""
    with open(output_srt_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments):
            start = seg['start']
            end = seg['end']
            text_en = seg['text'].strip()

            start_time = f"{int(start//3600):02d}:{int((start%3600)//60):02d}:{int(start%60):02d},{int((start%1)*1000):03d}"
            end_time   = f"{int(end//3600):02d}:{int((end%3600)//60):02d}:{int(end%60):02d},{int((end%1)*1000):03d}"

            text_zh = translate_text(text_en) if translate else ""

            f.write(f"{i+1}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text_en}\n")
            if translate and text_zh:
                f.write(f"{text_zh}\n")
            f.write("\n")

def burn_subtitles(video_path, srt_path, output_path):
    """使用 FFmpeg 烧录字幕到视频（自动创建输出目录）"""
    # 确保输出目录存在
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    video_dir = os.path.dirname(os.path.abspath(video_path))

    # 获取视频原始分辨率（用于字幕自适应）
    probe_cmd = [
        FFPROBE, '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height', '-of', 'json', video_path
    ]
    try:
        probe_bytes = subprocess.check_output(probe_cmd, stderr=subprocess.STDOUT)
        probe_out = probe_bytes.decode('utf-8', errors='ignore')
        info = json.loads(probe_out)
        original_size = f"{info['streams'][0]['width']}x{info['streams'][0]['height']}"
    except Exception:
        original_size = "1280x720"   # 回退

    # 将字幕文件复制到视频同目录（FFmpeg subtitles 滤镜要求相对路径或同目录）
    srt_filename = os.path.basename(srt_path)
    srt_in_video_dir = os.path.join(video_dir, srt_filename)

    need_cleanup = False
    if os.path.abspath(srt_path) != os.path.abspath(srt_in_video_dir):
        shutil.copy2(srt_path, srt_in_video_dir)
        need_cleanup = True

    # 根据操作系统选择字体（macOS 用 PingFang SC，Windows 用 SimHei）
    if sys.platform == "darwin":
        font_name = "PingFang SC"
    elif sys.platform == "win32":
        font_name = "SimHei"
    else:
        font_name = "Noto Sans CJK SC"

    # 构建字幕滤镜（黑底白字，中文字体）
    vf_filter = (
        f"subtitles={srt_filename}:"
        f"force_style='FontName={font_name},FontSize=24,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2':"
        f"original_size={original_size}"
    )

    cmd = [
        FFMPEG, '-i', video_path,
        '-vf', vf_filter,
        '-c:a', 'copy',
        '-y', output_path
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, cwd=video_dir, text=False)
        return True
    except subprocess.CalledProcessError as cpe:
        stderr = cpe.stderr.decode('utf-8', errors='ignore') if cpe.stderr else "未知错误"
        raise RuntimeError(f"FFmpeg 烧录字幕失败 (exit code {cpe.returncode}): {stderr}")
    finally:
        if need_cleanup and os.path.exists(srt_in_video_dir):
            os.remove(srt_in_video_dir)

# ========================= GUI 应用程序 ================================

class TranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("视频双语字幕生成器 v2.0")
        self.root.geometry("680x550")
        self.root.resizable(False, False)

        # 适配 Windows 高 DPI（macOS 不需要）
        if sys.platform == "win32":
            try:
                from ctypes import windll
                windll.shcore.SetProcessDpiAwareness(1)
            except:
                pass

        main_frame = ttk.Frame(root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # ----- 输入视频 -----
        ttk.Label(main_frame, text="输入视频:", font=("Microsoft YaHei", 10)).grid(row=0, column=0, sticky=tk.W, pady=5)
        self.input_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.input_var, width=50).grid(row=0, column=1, pady=5, padx=5)
        ttk.Button(main_frame, text="浏览...", command=self.browse_input).grid(row=0, column=2, pady=5)

        # ----- 输出目录 -----
        ttk.Label(main_frame, text="输出文件夹:", font=("Microsoft YaHei", 10)).grid(row=1, column=0, sticky=tk.W, pady=5)
        self.output_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.output_var, width=50).grid(row=1, column=1, pady=5, padx=5)
        ttk.Button(main_frame, text="浏览...", command=self.browse_output).grid(row=1, column=2, pady=5)

        # ----- 模型选择 -----
        ttk.Label(main_frame, text="Whisper模型:", font=("Microsoft YaHei", 10)).grid(row=2, column=0, sticky=tk.W, pady=5)
        self.model_var = tk.StringVar(value="base")
        model_combo = ttk.Combobox(main_frame, textvariable=self.model_var,
                                   values=["tiny", "base", "small", "medium", "large"],
                                   width=10, state="readonly")
        model_combo.grid(row=2, column=1, sticky=tk.W, pady=5, padx=5)
        ttk.Label(main_frame, text="(base 是速度与质量的平衡)", foreground="gray").grid(row=2, column=1, sticky=tk.E, pady=5)

        # ----- 启动按钮 -----
        self.start_btn = ttk.Button(main_frame, text="🚀 一键生成双语字幕视频", command=self.start_process)
        self.start_btn.grid(row=3, column=0, columnspan=3, pady=15)

        # ----- 进度条 -----
        self.progress = ttk.Progressbar(main_frame, length=600, mode='determinate')
        self.progress.grid(row=4, column=0, columnspan=3, pady=5)
        self.progress['value'] = 0

        # ----- 状态 -----
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(main_frame, textvariable=self.status_var, font=("Microsoft YaHei", 9), foreground="blue").grid(row=5, column=0, columnspan=3, pady=2)

        # ----- 日志文本框 -----
        ttk.Label(main_frame, text="运行日志:", font=("Microsoft YaHei", 9)).grid(row=6, column=0, sticky=tk.W, pady=(10,0))
        self.log_box = scrolledtext.ScrolledText(main_frame, width=75, height=15, state='disabled', wrap=tk.WORD)
        self.log_box.grid(row=7, column=0, columnspan=3, pady=5)

        # 日志队列轮询
        self.log_queue = queue.Queue()
        self.root.after(100, self.poll_log_queue)

        # 默认输出到桌面
        self.output_var.set(os.path.expanduser("~/Desktop"))

    def browse_input(self):
        path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv"), ("所有文件", "*.*")]
        )
        if path:
            self.input_var.set(path)
            self.output_var.set(os.path.dirname(path))

    def browse_output(self):
        path = filedialog.askdirectory(title="选择输出文件夹")
        if path:
            self.output_var.set(path)

    def log(self, msg):
        self.log_queue.put(msg)

    def poll_log_queue(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get()
            self.log_box.configure(state='normal')
            self.log_box.insert(tk.END, msg + "\n")
            self.log_box.see(tk.END)
            self.log_box.configure(state='disabled')
        self.root.after(100, self.poll_log_queue)

    def set_progress(self, value, status):
        self.progress['value'] = value
        self.status_var.set(status)
        self.root.update_idletasks()

    def start_process(self):
        video_file = self.input_var.get().strip()
        output_dir = self.output_var.get().strip()
        model_size = self.model_var.get()

        if not video_file or not os.path.exists(video_file):
            messagebox.showerror("错误", "请先选择有效的输入视频文件！")
            return

        if not output_dir:
            output_dir = os.path.dirname(video_file)
            self.output_var.set(output_dir)

        if os.path.isfile(output_dir):
            output_dir = os.path.dirname(output_dir)
            self.output_var.set(output_dir)

        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            messagebox.showerror("错误", f"无法创建输出文件夹：\n{e}")
            return

        base_name = os.path.splitext(os.path.basename(video_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}_双语字幕.mp4")

        if os.path.exists(output_file):
            if not messagebox.askyesno("文件已存在", f"输出文件已存在：\n{output_file}\n\n是否覆盖？"):
                return

        self.start_btn.configure(state='disabled')
        self.progress['value'] = 0
        self.log_box.configure(state='normal')
        self.log_box.delete(1.0, tk.END)
        self.log_box.configure(state='disabled')

        thread = threading.Thread(target=self.worker, args=(video_file, output_file, model_size), daemon=True)
        thread.start()

    def worker(self, video_file, output_file, model_size):
        error_msg = None
        try:
            self.log(f"📁 输入: {video_file}")
            self.log(f"💾 输出: {output_file}")
            self.log(f"🧠 Whisper模型: {model_size}")
            self.log("-" * 50)

            with tempfile.TemporaryDirectory() as tmpdir:
                audio_file = os.path.join(tmpdir, "audio.mp3")
                srt_file = os.path.join(tmpdir, "subtitles.srt")

                self.set_progress(10, "正在提取音频...")
                self.log("🎵 步骤 1/4: 提取音频...")
                extract_audio(video_file, audio_file)
                self.log("✅ 音频提取完成")

                self.set_progress(30, "正在进行语音识别，请耐心等待...")
                self.log("🎤 步骤 2/4: 加载 Whisper 模型并识别语音...")
                segments = transcribe_audio(audio_file, model_size)
                self.log(f"✅ 识别完成，共 {len(segments)} 个片段")

                self.set_progress(60, "正在翻译并生成字幕...")
                self.log("📝 步骤 3/4: 翻译并生成双语字幕...")
                create_subtitles(segments, srt_file, translate=True)
                self.log("✅ 字幕文件生成完成")

                self.set_progress(85, "正在将字幕烧录到视频中...")
                self.log("🔥 步骤 4/4: 使用 FFmpeg 烧录字幕到视频...")
                burn_subtitles(video_file, srt_file, output_file)
                self.log("✅ 字幕烧录完成")

            self.set_progress(100, "处理完成！")
            self.log("-" * 50)
            self.log("🎉 全部完成！")
            self.root.after(0, lambda: messagebox.showinfo("完成", f"视频已生成！\n\n保存位置：\n{output_file}"))

        except Exception as e:
            error_msg = str(e)
            self.log(f"❌ 错误: {error_msg}")
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("运行错误", msg))
        finally:
            self.root.after(0, lambda: self.start_btn.configure(state='normal'))
            self.root.after(0, lambda: self.status_var.set("就绪"))

# ========================= 入口 ========================================
def main():
    root = tk.Tk()
    app = TranslatorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
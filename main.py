import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import threading
import os

import torch
from diffusers import StableDiffusionImg2ImgPipeline

SD_MODEL_ID = "CompVis/stable-diffusion-v1-1"
IMAGE_PATH_IN = "input.png"
IMAGE_PATH_OUT = "output.png"

pipe = None
init_image_pil = None


def load_model():
    global pipe

    generate_btn.config(state="disabled")
    status_label.config(text="Загрузка Stable Diffusion v1.1...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            SD_MODEL_ID,
            torch_dtype=dtype,
            safety_checker=None,
        ).to(device)

        status_label.config(text=f"SD v1.1 img2img загружена ({device}, {dtype})")

    except Exception as e:
        pipe = None
        messagebox.showerror("Ошибка загрузки модели", str(e))
        status_label.config(text="Ошибка загрузки")


def choose_image():
    global init_image_pil

    file_path = filedialog.askopenfilename(
        title="Выберите картинку",
        filetypes=[("Изображения", "*.png *.jpg *.jpeg *.webp *.bmp")]
    )
    if not file_path:
        return

    try:
        img = Image.open(file_path).convert("RGB")
        img.save(IMAGE_PATH_IN)

        # превью
        preview = img.copy()
        preview.thumbnail((256, 256))
        tk_img = ImageTk.PhotoImage(preview)
        input_image_label.config(image=tk_img)
        input_image_label.image = tk_img

        init_image_pil = img
        status_label.config(text="Исходная картинка выбрана")

        if pipe is not None:
            generate_btn.config(state="normal")

    except Exception as e:
        messagebox.showerror("Ошибка открытия файла", str(e))


def generate_image():
    global pipe, init_image_pil

    if pipe is None:
        messagebox.showerror("Нет модели", "Модель ещё не загружена")
        return
    if init_image_pil is None:
        messagebox.showwarning("Нет картинки", "Сначала выбери исходное изображение")
        return

    generate_btn.config(state="disabled")
    status_label.config(text="Генерация...")

    # читаем значения слайдеров
    strength = strength_var.get()
    guidance = guidance_var.get()
    steps = int(steps_var.get())

    def worker():
        try:
            init_img = init_image_pil.resize((512, 512))
            prompt = ""  # без текста

            result = pipe(
                prompt=prompt,
                image=init_img,
                strength=strength,
                guidance_scale=guidance,
                num_inference_steps=steps,
            )
            out_image = result.images[0]
            out_image.save(IMAGE_PATH_OUT)
            show_output_image(IMAGE_PATH_OUT)
            status_label.config(
                text=f"Готово (strength={strength:.2f}, guidance={guidance:.1f}, steps={steps})"
            )

        except Exception as e:
            messagebox.showerror("Ошибка генерации", str(e))
            status_label.config(text="Ошибка генерации")
        finally:
            generate_btn.config(state="normal")

    threading.Thread(target=worker, daemon=True).start()


def show_output_image(path):
    img = Image.open(path)
    img.thumbnail((256, 256))
    tk_img = ImageTk.PhotoImage(img)
    output_image_label.config(image=tk_img)
    output_image_label.image = tk_img


def save_output_image():
    if not os.path.exists(IMAGE_PATH_OUT):
        messagebox.showwarning("Нет результата", "Сначала сгенерируй изображение")
        return

    file_path = filedialog.asksaveasfilename(
        title="Сохранить результат",
        initialfile="output.png",
        defaultextension=".png",
        filetypes=[
            ("PNG", "*.png"),
            ("JPEG", "*.jpg *.jpeg"),
            ("WebP", "*.webp"),
        ],
    )
    if not file_path:
        return

    try:
        Image.open(IMAGE_PATH_OUT).convert("RGB").save(file_path)
        status_label.config(text=f"Результат сохранён: {file_path}")
    except Exception as e:
        messagebox.showerror("Ошибка сохранения", str(e))


def delete_images():
    for p in (IMAGE_PATH_IN, IMAGE_PATH_OUT):
        if os.path.exists(p):
            os.remove(p)
    input_image_label.config(image="")
    output_image_label.config(image="")
    status_label.config(text="Файлы удалены")


root = tk.Tk()
root.title("Бэкрумзинатор")
root.geometry("800x650")

frame_top = ttk.Frame(root, padding=10)
frame_top.pack(fill="x")

choose_btn = ttk.Button(frame_top, text="Выбрать картинку", command=choose_image)
choose_btn.pack(side="left", padx=5)

generate_btn = ttk.Button(frame_top, text="Перерисовать", command=generate_image, state="disabled")
generate_btn.pack(side="left", padx=5)

delete_btn = ttk.Button(frame_top, text="Удалить файлы", command=delete_images)
delete_btn.pack(side="left", padx=5)

save_btn = ttk.Button(frame_top, text="Сохранить результат", command=save_output_image)
save_btn.pack(side="left", padx=5)

params_frame = ttk.LabelFrame(root, text="Параметры генерации", padding=10)
params_frame.pack(fill="x", padx=10, pady=5)

strength_var = tk.DoubleVar(value=0.6)
guidance_var = tk.DoubleVar(value=7.5)
steps_var = tk.DoubleVar(value=50)

ttk.Label(params_frame, text="Strength (0–1)").grid(row=0, column=0, sticky="w")
strength_scale = ttk.Scale(
    params_frame, from_=0.0, to=1.0, orient="horizontal", variable=strength_var
)
strength_scale.grid(row=0, column=1, sticky="ew", padx=5)

ttk.Label(params_frame, text="Guidance scale").grid(row=1, column=0, sticky="w")
guidance_scale = ttk.Scale(
    params_frame, from_=1.0, to=15.0, orient="horizontal", variable=guidance_var
)
guidance_scale.grid(row=1, column=1, sticky="ew", padx=5)

ttk.Label(params_frame, text="Steps (проходы)").grid(row=2, column=0, sticky="w")
steps_scale = ttk.Scale(
    params_frame, from_=10, to=100, orient="horizontal", variable=steps_var
)
steps_scale.grid(row=2, column=1, sticky="ew", padx=5)

params_frame.columnconfigure(1, weight=1)

frame_images = ttk.Frame(root, padding=10)
frame_images.pack(fill="both", expand=True)

ttk.Label(frame_images, text="Исходное").grid(row=0, column=0, pady=5)
ttk.Label(frame_images, text="Результат").grid(row=0, column=1, pady=5)

input_image_label = tk.Label(frame_images)
input_image_label.grid(row=1, column=0, padx=10, pady=10)

output_image_label = tk.Label(frame_images)
output_image_label.grid(row=1, column=1, padx=10, pady=10)

status_label = ttk.Label(root, text="Загрузка модели...")
status_label.pack(pady=5)

threading.Thread(target=load_model, daemon=True).start()

root.mainloop()

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST

from .model_inference import predict_restrictions
from .models import Analysis


def home(request):
    return render(request, "analyzer/index.html")


@require_POST
def analyze(request):
    text = request.POST.get("text", "").strip()
    image = request.FILES.get("image")
    if image:
        record = Analysis.objects.create(image=image)
        if not text:
            try:
                from .ocr import extract_text
                text = extract_text(record.image.path)
            except Exception as exc:
                return JsonResponse({"error": f"OCR could not read this image: {exc}", "hint": "Check the image is clear, or paste the ingredients manually."}, status=422)
    elif not text:
        return JsonResponse({"error": "Upload a label image or paste ingredient text."}, status=400)
    else:
        record = Analysis.objects.create()
    selected_restrictions = request.POST.getlist("restrictions")
    dietary_analysis, model_errors = predict_restrictions(text, selected_restrictions)
    if dietary_analysis is None:
        return JsonResponse({"error": "The final trained model is unavailable.", "model_errors": model_errors}, status=503)
    result = {"models": {}, "model_errors": model_errors, "analysis": dietary_analysis}
    record.raw_text = text
    record.ingredients = [part.strip() for part in text.replace("\n", ",").split(",") if part.strip()]
    record.restrictions = selected_restrictions
    record.result = result
    record.save(update_fields=["raw_text", "ingredients", "restrictions", "result"])
    return JsonResponse({"id": record.id, "ocr_text": text, **result})

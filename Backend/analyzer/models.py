from django.db import models


class Analysis(models.Model):
    image = models.ImageField(upload_to="labels/", blank=True, null=True)
    raw_text = models.TextField(blank=True)
    ingredients = models.JSONField(default=list)
    restrictions = models.JSONField(default=list)
    result = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

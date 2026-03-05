from django.contrib.auth.models import AbstractUser
from django.db import models

class User(AbstractUser):
    email = models.EmailField(unique=True)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']

    class Meta:
        db_table = 'user' 
        
    def __str__(self):
        return self.email

class Dataset(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file_hash = models.CharField(max_length=64, null=True, blank=True)
    cleaned_data = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "datasets"
        
    def __str__(self):
        return f"Dataset {self.id} by {self.user.email}"
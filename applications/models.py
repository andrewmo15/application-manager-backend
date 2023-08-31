from django.db import models

# Create your models here.
class Application(models.Model):
    user_id = models.CharField(max_length=100)
    company = models.CharField(max_length=100)
    status = models.CharField(max_length=100)
    position = models.CharField(max_length = 100)

class Users(models.Model):
    email = models.CharField(max_length=100)
    last_refresh = models.CharField(max_length=100)
    email_type = models.CharField(max_length=100)
from django.db import models

# Create your models here.
class Application(models.Model):
    username = models.CharField(max_length=100)
    company = models.CharField(max_length=100)
    status = models.CharField(max_length=100)
    position = models.CharField(max_length = 100)

class UserInfo(models.Model):
    username = models.CharField(max_length=100)
    email = models.CharField(max_length=100)
    last_refresh = models.CharField(max_length=100)
    imap_password = models.CharField(max_length=100)
    imap_url = models.CharField(max_length=100)
# Generated by Django 4.1.4 on 2023-03-26 02:28

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('applications', '0002_application_username'),
    ]

    operations = [
        migrations.CreateModel(
            name='UserInfo',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('username', models.CharField(max_length=100)),
                ('first_name', models.CharField(max_length=100)),
                ('last_name', models.CharField(max_length=100)),
                ('last_refresh', models.DateTimeField(max_length=100)),
                ('imap_password', models.CharField(max_length=100)),
                ('imap_url', models.CharField(max_length=100)),
            ],
        ),
    ]

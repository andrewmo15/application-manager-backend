# Generated by Django 4.1.4 on 2023-03-29 19:17

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('applications', '0004_userinfo_email_alter_userinfo_last_refresh'),
    ]

    operations = [
        migrations.AlterField(
            model_name='userinfo',
            name='email',
            field=models.CharField(max_length=100),
        ),
    ]

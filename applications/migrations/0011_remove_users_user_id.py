# Generated by Django 4.1.4 on 2023-08-31 01:32

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('applications', '0010_rename_email_application_user_id_users_user_id'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='users',
            name='user_id',
        ),
    ]
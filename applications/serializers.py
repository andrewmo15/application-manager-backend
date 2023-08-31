from rest_framework import serializers
from .models import Application, Users

class ApplicationSerializers(serializers.ModelSerializer):
    class Meta:
        model = Application
        fields = '__all__'
    
class UsersSerializers(serializers.ModelSerializer):
    class Meta:
        model = Users
        fields = '__all__'
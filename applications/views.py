from .models import Application, Users
from .serializers import ApplicationSerializers, UsersSerializers
from rest_framework import status, viewsets
from rest_framework.response import Response
from .gmailparser import GmailParser
from .classifier import EmailData
from django.shortcuts import get_object_or_404
from django.http import HttpResponse
import json

class ApplicationViewSet(viewsets.ViewSet):

    # GET /applications
    def list(self, request):
        applications = Application.objects.all()
        serializer = ApplicationSerializers(applications, many=True)
        return Response(serializer.data)

    # POST /applications
    def create(self, request):
        queryset = Application.objects.filter(user_id=request.data["user_id"], company=request.data["company"], position=request.data["position"])
        if not queryset.exists():
            serializer = ApplicationSerializers(data=request.data)
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data, status=status.HTTP_201_CREATED)
            else:
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        return self.update(request, queryset.first().id)

    # GET /applications/id
    def retrieve(self, request, pk=None):
        queryset = Application.objects.all()
        application = get_object_or_404(queryset, pk=pk)
        serializer = ApplicationSerializers(application)
        return Response(serializer.data, status=status.HTTP_200_OK)

    # PUT /applications/id
    def update(self, request, pk=None):
        application = Application.objects.get(pk=pk)
        serializer = ApplicationSerializers(application, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    # DELETE /applications/id
    def destroy(self, request, pk=None):
        application = Application.objects.get(pk=pk)
        application.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class UsersViewSet(viewsets.ViewSet):

    # GET /users
    def list(self, requests):
        users = Users.objects.all()
        serializer = UsersSerializers(users, many=True)
        return Response(serializer.data)

    # POST /users
    def create(self, request):
        queryset = Users.objects.filter(email=request.data["email"])
        if not queryset.exists():
            serializer = UsersSerializers(data=request.data)
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data, status=status.HTTP_201_CREATED)
            else:
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        dic = queryset.first().__dict__
        del dic['_state']
        return HttpResponse(json.dumps(dic))

    # GET /users/id
    def retrieve(self, request, pk=None):
        queryset = Users.objects.all()
        user = get_object_or_404(queryset, pk=pk)
        serializer = UsersSerializers(user)
        return Response(serializer.data, status=status.HTTP_200_OK)

    # PUT /users/id
    def update(self, request, pk=None):
        user = Users.objects.get(pk=pk)
        serializer = UsersSerializers(user, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    # DELETE /users/id
    def destroy(self, request, pk=None):
        users = Users.objects.get(pk=pk)
        users.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

# GET newApplications/id
def getNewApplications(request, pk=None):
    token = request.META["HTTP_AUTHORIZATION"]
    try:
        user = Users.objects.filter(pk=pk)[0]
    except:
        return Response(status=status.HTTP_400_BAD_REQUEST)
    if request.method == "GET":
        last_refresh = user.last_refresh
        email_type = user.email_type
        emails = []
        if email_type == "gmail":
            parser = GmailParser(token, last_refresh)
            emails = parser.getEmails()
        classifier = EmailData(emails)
        newApplications = classifier.getClassifications()
        return HttpResponse(json.dumps({'newApplications': newApplications}))
    return Response(status=status.HTTP_400_BAD_REQUEST)

# GET userApplications/id
def getUserApplications(request, user_id=None):
    if request.method == "GET":
        queryset = Application.objects.filter(user_id=user_id)
        rtn = [ApplicationSerializers(query).data for query in queryset]
        return HttpResponse(json.dumps({"applications": rtn}))
    return Response(status=status.HTTP_400_BAD_REQUEST)

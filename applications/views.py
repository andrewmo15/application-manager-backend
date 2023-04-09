from .models import Application, UserInfo
from .serializers import ApplicationSerializers, UserSerializers, UserInfoSerializers
from rest_framework import status, viewsets
from rest_framework.response import Response
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from django.contrib.auth.models import User
from .emailparser import EmailParser
from .classifier import EmailData
from rest_framework.decorators import permission_classes, authentication_classes
from django.http import JsonResponse, HttpResponse
from django.shortcuts import get_object_or_404

class ApplicationViewSet(viewsets.ViewSet):
    permission_classes = [IsAuthenticated]
    authentication_classes = (TokenAuthentication,)

    # GET /applications
    def list(self, request):
        applications = Application.objects.all()
        serializer = ApplicationSerializers(applications, many=True)
        return Response(serializer.data)
    
    # POST /applications
    def create(self, request):
        serializer = ApplicationSerializers(data=request.data)
        if serializer.is_valid():
            queryset = Application.objects.filter(username=request.data["username"], company=request.data["company"], position=request.data["position"])
            if not queryset.exists():
                serializer.save()
            else:
                self.update(request, queryset.first().id)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    # GET /applications/id
    def retrieve(self, request, pk=None):
        queryset = Application.objects.all()
        application = get_object_or_404(queryset, pk=pk)
        serializer = ApplicationSerializers(application)
        return JsonResponse(serializer.data)

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

class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializers

class UserInfoViewSet(viewsets.ViewSet):
    permission_classes = [IsAuthenticated]
    authentication_classes = (TokenAuthentication,)

    # GET /userinfo
    def list(self, request):
        userinfos = UserInfo.objects.all()
        serializer = UserInfoSerializers(userinfos, many=True)
        return Response(serializer.data)
    
    # POST /userinfo
    def create(self, request):
        serializer = UserInfoSerializers(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    # GET /userinfo/username
    def retrieve(self, request, pk=None):
        queryset = UserInfo.objects.filter(username=pk)
        serializer = UserInfoSerializers(queryset.first())
        qsetuser = User.objects.filter(username=pk)
        user = qsetuser.first()
        serializer_data = serializer.data
        serializer_data["user_id"] = user.id
        return Response(serializer_data)

    # PUT /userinfo/username
    def update(self, request, pk=None):
        userinfo = UserInfo.objects.get(username=pk)
        serializer = UserInfoSerializers(userinfo, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    # DELETE /userinfo/username
    def destroy(self, request, pk=None):
        userinfo = UserInfo.objects.get(username=pk)
        userinfo.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
    
@permission_classes([IsAuthenticated])
@authentication_classes(TokenAuthentication,)
# GET newApplications/username
def getNewApplications(request, username=None):
    try:
        user = UserInfo.objects.filter(username=username)[0].email
    except:
        return HttpResponse(status=404)
    if request.method == "GET":
        try:
            userinfo = UserInfo.objects.filter(username=username)[0]
            password = userinfo.imap_password
            imap_url = userinfo.imap_url
            last_refresh = userinfo.last_refresh
        except:
            return HttpResponse(status=404)
        parser = EmailParser(user, password, imap_url, last_refresh)
        emails = parser.getEmails()
        classifier = EmailData(emails)
        newApplications = classifier.getClassifications()
        return JsonResponse({'newApplications': newApplications})
    return HttpResponse(status=400)

@permission_classes([IsAuthenticated])
@authentication_classes(TokenAuthentication,)
# GET userApplications/username
def getUserApplications(request, username=None):
    queryset = Application.objects.filter(username=username)
    rtn = [ApplicationSerializers(query).data for query in queryset]
    return JsonResponse({"applications": rtn})

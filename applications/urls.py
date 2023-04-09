from django.urls import path, include
from .views import ApplicationViewSet, UserViewSet, UserInfoViewSet, getNewApplications, getUserApplications
from rest_framework.routers import DefaultRouter

router = DefaultRouter()
router.register('applications', ApplicationViewSet, basename='applications')
router.register('users', UserViewSet, basename='users')
router.register('userinfo', UserInfoViewSet, basename="userinfo")
urlpatterns = [
    path('', include(router.urls)),
    path('newApplications/<str:username>', getNewApplications),
    path('userApplications/<str:username>', getUserApplications)
]

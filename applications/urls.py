from django.urls import path, include
from .views import ApplicationViewSet, UsersViewSet, getNewApplications, getUserApplications
from rest_framework.routers import DefaultRouter

router = DefaultRouter()
router.register('applications', ApplicationViewSet, basename='applications')
router.register('users', UsersViewSet, basename='users')
urlpatterns = [
    path('', include(router.urls)),
    path('newApplications/<str:pk>', getNewApplications),
    path('userApplications/<str:user_id>', getUserApplications)
]

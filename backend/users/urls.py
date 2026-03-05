from django.urls import path
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

from .views import (
    RegisterView,
    ProfileView,
    AdminUserListView,
    AdminDashboardView,
    UploadDatasetView,
    UserDatasetListView,
    DatasetDetailView,
)

urlpatterns = [
    # Auth
    path("register/", RegisterView.as_view()),
    path("login/", TokenObtainPairView.as_view()),
    path("refresh/", TokenRefreshView.as_view()),

    # Normal user
    path("profile/", ProfileView.as_view()),
    path("datasets/upload/", UploadDatasetView.as_view()),
    path("datasets/", UserDatasetListView.as_view()),
    path("datasets/<int:pk>/", DatasetDetailView.as_view()),

    # Admin-only
    path("admin/users/", AdminUserListView.as_view()),
    path("admin/dashboard/", AdminDashboardView.as_view()),
]
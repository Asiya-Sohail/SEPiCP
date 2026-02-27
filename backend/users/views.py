# from django.shortcuts import render
# from rest_framework.decorators import api_view

from rest_framework import generics, permissions
from rest_framework.response import Response
from rest_framework.views import APIView
from django.contrib.auth import get_user_model
from .serializers import RegisterSerializer, UserSerializer
from rest_framework.permissions import IsAdminUser, IsAuthenticated

User = get_user_model()


# 1️⃣ Register View
class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = RegisterSerializer


# 2️⃣ Get Logged-in User Profile
class ProfileView(generics.RetrieveAPIView):
    serializer_class = UserSerializer
    permission_classes = [IsAuthenticated]

    def get_object(self):
        return self.request.user


# 3️⃣ Admin - View All Users
class AdminUserListView(generics.ListAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [IsAdminUser]


# 4️⃣ Admin Dashboard Stats (Optional but Professional)
class AdminDashboardView(APIView):
    permission_classes = [IsAdminUser]

    def get(self, request):
        total_users = User.objects.count()
        admin_users = User.objects.filter(is_staff=True).count()

        return Response({
            "total_users": total_users,
            "admin_users": admin_users,
        })

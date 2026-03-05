# Validator + converter
from rest_framework import serializers
from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from .models import Dataset

User = get_user_model()


class RegisterSerializer(serializers.ModelSerializer):
    """
    Handles registration of a new user account.
    - Ensures email is unique.
    - Validates password strength.
    - Creates a normal (non-admin) user.
    """

    password = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = ["id", "username", "email", "password", "first_name"]
    # ✅ Custom validation for username
    def validate_username(self, value):
        if " " in value:
            raise serializers.ValidationError("Username cannot contain spaces.")
        return value

    def validate_email(self, value: str) -> str:
        if User.objects.filter(email=value).exists():
            raise serializers.ValidationError("Email already exists")
        return value

    def validate_password(self, value: str) -> str:
        validate_password(value)
        return value

    def create(self, validated_data):
        # Always create a normal user from this endpoint
        user = User.objects.create_user(
            username=validated_data["username"],
            email=validated_data["email"],
            password=validated_data["password"],
            first_name=validated_data.get("first_name", "")
        )
        user.is_staff = False
        user.is_superuser = False
        user.save(update_fields=["is_staff", "is_superuser"])
        return user


class UserSerializer(serializers.ModelSerializer):
    """
    Read-only representation of a user for profile and admin views.
    """

    class Meta:
        model = User
        fields = ["id", "username", "email", "is_staff", "is_superuser", "date_joined"]
        read_only_fields = ["id", "email", "is_staff", "is_superuser", "date_joined"]

class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = ['id', 'cleaned_data', 'created_at']
        read_only_fields = ['cleaned_data', 'created_at']
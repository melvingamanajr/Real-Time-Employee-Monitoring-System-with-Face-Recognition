import base64
from django import template

register = template.Library()


@register.filter
def base64_encode(value):
    encoded_bytes = base64.b64encode(value)
    encoded_string = encoded_bytes.decode('utf-8')
    return encoded_string

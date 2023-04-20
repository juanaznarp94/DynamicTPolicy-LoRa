import cv2


def plot_eclipses():
    image = cv2.imread('gps.png')
    overlay = image.copy()

    # Window name in which image is displayed
    window_name = 'Image'

    center_coordinates = (375, 255)
    center_coordinates_2 = (375, 298)
    center_coordinates_3 = (375, 341)
    center_coordinates_4 = (375, 384)
    center_coordinates_5 = (375, 427)
    center_coordinates_6 = (375, 470)
    center_coordinates_7 = (375, 513)
    center_coordinates_8 = (375, 556)

    axesLength = (42, 30)
    axesLength_2 = (85, 60)
    axesLength_3 = (120, 90)
    axesLength_4 = (165, 120)
    axesLength_5 = (208, 130)
    axesLength_6 = (230, 150)
    axesLength_7 = (280, 170)
    axesLength_8 = (323, 190)

    angle = 90
    startAngle = 90
    endAngle = 450

    color = (127, 255, 0)
    color_2 = (0, 255, 0)
    color_3 = (50, 205, 50)
    color_4 = (0, 128, 0)
    color_5 = (37, 73, 141)
    color_6 = (0, 0, 255)
    color_7 = (0, 0, 200)
    color_8 = (37, 73, 141)

    # Line thickness of 5 px
    thickness = -1
    alpha = 0.75
    # Using cv2.ellipse() method
    # Draw a ellipse with red line borders of thickness of 5 px
    cv2.ellipse(image, center_coordinates_8, axesLength_8,
                angle, startAngle, endAngle, color_8, thickness)
    cv2.ellipse(image, center_coordinates_7, axesLength_7,
                angle, startAngle, endAngle, color_7, thickness)
    cv2.ellipse(image, center_coordinates_6, axesLength_6,
                angle, startAngle, endAngle, color_6, thickness)
    cv2.ellipse(image, center_coordinates_5, axesLength_5,
                angle, startAngle, endAngle, color_5, thickness)
    cv2.ellipse(image, center_coordinates_4, axesLength_4,
                angle, startAngle, endAngle, color_4, thickness)
    cv2.ellipse(image, center_coordinates_3, axesLength_3,
                angle, startAngle, endAngle, color_3, thickness)
    cv2.ellipse(image, center_coordinates_2, axesLength_2,
                angle, startAngle, endAngle, color_2, thickness)
    cv2.ellipse(image, center_coordinates, axesLength,
                angle, startAngle, endAngle, color, thickness)

    image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Displaying the image
    cv2.imshow(window_name, image_new)
    cv2.waitKey()
    cv2.imwrite('map_modified_eclipses.png', image_new)

def plot_points():
    image = cv2.imread('map_modified_eclipses.png')
    overlay = image.copy()
    window_name = 'Image'

    start_point = (367, 210)
    end_point = (376, 219)
    color = (0, 0, 0)
    thickness = -1

    img = cv2.rectangle(image, start_point, end_point, color, thickness)

    # img = cv2.circle(image, (372, 215), 5, (0, 0, 0), -1)

    img = cv2.circle(image, (372, 299), 4, (0, 0, 0), -1)
    img = cv2.circle(image, (372, 380), 4, (0, 0, 0), -1)
    img = cv2.circle(image, (386, 460), 4, (0, 0, 0), -1)
    img = cv2.circle(image, (395, 545), 4, (0, 0, 0), -1)
    img = cv2.circle(image, (395, 545), 4, (0, 0, 0), -1)
    img = cv2.circle(image, (395, 631), 4, (0, 0, 0), -1)
    img = cv2.circle(image, (377, 700), 4, (0, 0, 0), -1)
    img = cv2.circle(image, (432, 776), 4, (0, 0, 0), -1)
    img = cv2.circle(image, (443, 858), 4, (0, 0, 0), -1)

    alpha = 0.5

    image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    cv2.imshow(window_name, image_new)
    cv2.waitKey()
    cv2.imwrite('map_modified_eclipses_points.png', image_new)

plot_eclipses()
#plot_points()
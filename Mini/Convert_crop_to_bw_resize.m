%imshow(a);
dinfo = dir('Crop_4/*.jpeg');
disp(length(dinfo));
for K =1:length(dinfo)
      %disp(K);
      thisdata=sprintf('Crop_4/xyz%d.jpeg',K);
      a=imread(thisdata);
      a=im2bw(a);
      a=imresize(a,[50,50]);
      n=sprintf('Last_4/4_(%d).jpeg',K);
      imwrite(a,n);
end
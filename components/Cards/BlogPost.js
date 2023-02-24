import React from 'react';
import PropTypes from 'prop-types';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import Paper from '@mui/material/Paper';
import { useTranslation } from 'next-i18next';
import { useText } from '/theme/common';
import useStyles from './cards-style';

function BlogPost(props) {
  const { classes } = useStyles();
  const { classes: text } = useText();
  const { t } = useTranslation('common');
  const {
    img,
    title,
    desc,
  } = props;

  return (
    <Paper className={classes.post}>
      <figure>
        <img src={img} alt="thumb" />
      </figure>
      <div className={classes.text}>
        <Typography variant="h5" className={text.subtitle2}>{title}</Typography>
        <Typography display="block" component="p" className={text.paragraph}>{desc}</Typography>
      </div>
      <Button
        href="#"
        color="primary"
        className={classes.readmore}
        classes={{
          root: classes.rootReadmore,
          text: classes.textReadmore
        }}
      >
        {t('profile-landing.read_more')}
      </Button>
    </Paper>
  );
}

BlogPost.propTypes = {
  img: PropTypes.string.isRequired,
  title: PropTypes.string.isRequired,
  desc: PropTypes.string.isRequired,
};

export default BlogPost;
